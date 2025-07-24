import torch
from maskrcnn_rui.modeling.make_layers import make_conv3x3
from maskrcnn_rui.modeling.make_layers import make_fc
from torch.nn import functional as F

from .inference import make_roi_keypoint_post_processor
from .loss import make_roi_keypoint_loss_evaluator
from .roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from .roi_keypoint_predictors import make_roi_keypoint_predictor


class ROIKeypointHead(torch.nn.Module):
    def __init__(self, cfg, opt, in_channels, if_roi_h_heads=False):
        super().__init__()
        self.cfg = cfg.clone()
        self.opt = opt
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_keypoint_predictor(
            cfg,
            self.feature_extractor.out_channels,
        )
        self.post_processor = make_roi_keypoint_post_processor()
        self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)

        self.if_roi_h_heads = if_roi_h_heads
        if self.if_roi_h_heads:
            use_gn = False
            input_size = 256 * 7 * 7
            representation_size = 1024
            final_size = self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES_h

            # v1
            kaiming_init = True  # always yes
            self.predictor_person_h_conv33 = make_conv3x3(
                self.feature_extractor.out_channels,
                256,
                stride=2,
                use_relu=True,
                kaiming_init=kaiming_init,
            )
            self.predictor_person_h_fc6 = make_fc(
                input_size,
                representation_size,
                use_gn,
                kaiming_init=kaiming_init,
            )
            self.predictor_person_h_fc7 = make_fc(
                representation_size,
                final_size,
                use_gn,
                kaiming_init=kaiming_init,
            )

    def forward(
        self,
        features,
        proposals,
        targets=None,
        target_idxes_with_valid_kps_list=[],
        if_notNMS_yet=True,
    ):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # if self.training:
        with torch.no_grad():
            if self.opt.est_bbox:
                if if_notNMS_yet:
                    proposals = self.loss_evaluator.subsample(
                        proposals,
                        targets,
                        if_sample=True,
                    )
            else:
                proposals_valid = [
                    proposal[target_idxes_with_valid_kps]
                    for proposal, target_idxes_with_valid_kps in zip(
                        proposals,
                        target_idxes_with_valid_kps_list,
                    )
                ]
                targets_valid = [
                    target[target_idxes_with_valid_kps]
                    for target, target_idxes_with_valid_kps in zip(
                        targets,
                        target_idxes_with_valid_kps_list,
                    )
                ]
                proposals_valid = self.loss_evaluator.prepare_targets_for_gt_box_input(
                    proposals_valid,
                    targets_valid,
                )
        if not self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = self.feature_extractor(features, proposals)
        else:
            x = features
        kp_logits = self.predictor(x)

        output_kp = {}
        if self.if_roi_h_heads and (not if_notNMS_yet):
            # v1
            x_conv = self.predictor_person_h_conv33(x)
            person_h_logits = x_conv.view(x_conv.size(0), -1)
            person_h_logits = F.relu(self.predictor_person_h_fc6(person_h_logits))
            person_h_logits = self.predictor_person_h_fc7(person_h_logits)
            output_kp.update({"person_h_logits": person_h_logits})

        proposals_post = None
        loss_kp = None
        if self.opt.est_bbox:
            if not if_notNMS_yet:
                proposals_post = self.post_processor(kp_logits, proposals)
            else:
                loss_kp = self.loss_evaluator(proposals, kp_logits)
        else:
            num_bbox_per_image = [proposal.bbox.shape[0] for proposal in proposals]
            num_bbox_valid_per_image = [
                proposal_valid.bbox.shape[0] for proposal_valid in proposals_valid
            ]
            assert sum(num_bbox_per_image) == kp_logits.shape[0], "%d-%d" % (
                sum(num_bbox_per_image),
                kp_logits.shape[0],
            )
            kp_logits_list = kp_logits.split(num_bbox_per_image)
            kp_logits_valid_list = [
                kp_logit[target_idxes_with_valid_kps]
                for kp_logit, target_idxes_with_valid_kps in zip(
                    kp_logits_list,
                    target_idxes_with_valid_kps_list,
                )
            ]
            kp_logits_valid = torch.cat(kp_logits_valid_list)
            assert (
                sum(num_bbox_valid_per_image) == kp_logits_valid.shape[0]
            ), "%d-%d" % (sum(num_bbox_valid_per_image), kp_logits_valid.shape[0])
            loss_kp = self.loss_evaluator(proposals_valid, kp_logits_valid)
        return x, proposals, proposals_post, dict(loss_kp=loss_kp), output_kp


def build_roi_keypoint_head(cfg, opt, in_channels, if_roi_h_heads=False):
    return ROIKeypointHead(cfg, opt, in_channels, if_roi_h_heads=if_roi_h_heads)
