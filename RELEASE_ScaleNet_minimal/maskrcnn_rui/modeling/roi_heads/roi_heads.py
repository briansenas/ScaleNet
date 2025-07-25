# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .box_head.box_head import build_roi_box_head_rui
from .box_head.roi_box_predictors import make_roi_box_predictor_rui
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .mask_head.mask_head import build_roi_mask_head


# ============ roi_h heads ===============
class CombinedROIHeadsRui(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super().__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, features, proposals, targets=None):
        losses = {}
        loss_box = self.person_h(features, proposals, targets)
        losses.update(loss_box)
        return losses


def build_roi_h_heads(cfg, opt, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(
            (
                "person_h",
                build_roi_box_head_rui(cfg, in_channels, make_roi_box_predictor_rui),
            ),
        )

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeadsRui(cfg, roi_heads)

    return roi_heads


# ============ classifier heads ===============
class CombinedClassifierHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, opt, heads):
        super().__init__(heads)
        self.cfg = cfg.clone()
        self.opt = opt

    def forward(self, features, proposals, targets=None):
        losses = {}
        loss_horizon = self.classifier_horizon(features, proposals, targets)
        loss_pitch = self.classifier_pitch(features, proposals, targets)
        loss_roll = self.classifier_roll(features, proposals, targets)
        loss_vfov = self.classifier_vfov(features, proposals, targets)
        losses.update(
            {
                "output_horizon": loss_horizon,
                "output_pitch": loss_pitch,
                "output_roll": loss_roll,
                "output_vfov": loss_vfov,
            },
        )
        if not self.opt.pointnet_camH:
            loss_camH = self.classifier_camH(features, proposals, targets)
            losses.update({"output_camH": loss_camH})
        return losses


def build_classifier_heads(cfg, opt, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    roi_heads.append(
        (
            "classifier_horizon",
            build_roi_box_head_rui(cfg, in_channels, make_roi_box_predictor_rui),
        ),
    )
    roi_heads.append(
        (
            "classifier_pitch",
            build_roi_box_head_rui(cfg, in_channels, make_roi_box_predictor_rui),
        ),
    )
    roi_heads.append(
        (
            "classifier_roll",
            build_roi_box_head_rui(cfg, in_channels, make_roi_box_predictor_rui),
        ),
    )
    roi_heads.append(
        (
            "classifier_vfov",
            build_roi_box_head_rui(cfg, in_channels, make_roi_box_predictor_rui),
        ),
    )
    if not opt.pointnet_camH:
        if opt.direct_h:
            roi_heads.append(
                (
                    "classifier_camH",
                    build_roi_box_head_rui(
                        cfg,
                        in_channels,
                        make_roi_box_predictor_rui,
                        output_cls_num=1,
                    ),
                ),
            )
        else:
            roi_heads.append(
                (
                    "classifier_camH",
                    build_roi_box_head_rui(
                        cfg,
                        in_channels,
                        make_roi_box_predictor_rui,
                    ),
                ),
            )
        pass

    # combine individual heads in a single module
    roi_heads = CombinedClassifierHeads(cfg, opt, roi_heads)

    return roi_heads


# ============ original bbox heads ===============
class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super().__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if (
            cfg.MODEL.KEYPOINT_ON
            and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
        ):
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(
        self,
        opt,
        features,
        proposals,
        targets=None,
        target_idxes_with_valid_kps_list=[],
    ):
        losses = {}
        outputs = {}
        if opt.est_bbox:
            x, detections, detections_nms, loss_box = self.box(
                features,
                proposals,
                targets=targets,
            )
            losses.update(loss_box)
        else:
            detections = proposals
            detections_nms = proposals

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON and opt.est_kps:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            if opt.est_bbox:
                x, detections, _, loss_keypoint, _ = self.keypoint(
                    keypoint_features,
                    detections,
                    targets,
                    target_idxes_with_valid_kps_list=target_idxes_with_valid_kps_list,
                    if_notNMS_yet=True,
                )
                _, _, detections_nms, _, output_kp = self.keypoint(
                    keypoint_features,
                    detections_nms,
                    targets,
                    target_idxes_with_valid_kps_list=target_idxes_with_valid_kps_list,
                    if_notNMS_yet=False,
                )
            else:
                x, detections, detections_nms, loss_keypoint, output_kp = self.keypoint(
                    keypoint_features,
                    detections,
                    targets,
                    target_idxes_with_valid_kps_list=target_idxes_with_valid_kps_list,
                )
            losses.update(loss_keypoint)
            outputs.update(output_kp)

        return x, detections, detections_nms, losses, outputs


def build_roi_bbox_heads(cfg, opt, in_channels, if_roi_h_heads=False):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY and opt.est_bbox:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON and opt.est_kps:
        roi_heads.append(
            (
                "keypoint",
                build_roi_keypoint_head(
                    cfg,
                    opt,
                    in_channels,
                    if_roi_h_heads=if_roi_h_heads,
                ),
            ),
        )
    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
