# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor


class ROIBoxHeadRui(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels, predictor_fn, output_cls_num=None):
        super().__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)

        self.predictor = predictor_fn(
            cfg,
            self.feature_extractor.out_channels,
            output_cls_num=output_cls_num,
        )
        # self.post_processor = make_roi_box_post_processor(cfg)
        # self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    # In detectron 2 should accept (images, features, proposals, gt_instances | None)
    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # Can just delete images + gt_instances if not using them.
        # The problem is, we must measure the loss here, so maybe gt_instances should contain gt_logits
        x = self.feature_extractor(features, proposals)
        class_logits = self.predictor(x)
        return {"class_logits": class_logits}


def build_roi_box_head_rui(cfg, in_channels, predictor_fn, output_cls_num=None):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    # return ROIBoxHead(cfg, in_channels)
    return ROIBoxHeadRui(cfg, in_channels, predictor_fn, output_cls_num=output_cls_num)


# ======= original ==========
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super().__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg,
            self.feature_extractor.out_channels,
        )
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if targets is not None:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits],
            [box_regression],
        )

        proposals_nms = self.post_processor((class_logits, box_regression), proposals)
        return (
            x,
            proposals,
            proposals_nms,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
