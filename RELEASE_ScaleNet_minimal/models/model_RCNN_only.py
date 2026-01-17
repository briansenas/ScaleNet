import torch
import torch.nn as nn
from termcolor import colored
from utils.checkpointer import DetectronCheckpointer
from utils.utils_misc import green
from utils.utils_misc import white_blue

from .model_part_GeneralizedRCNNRuiMod_cameraCalib_sep import (
    GeneralizedRCNNRuiMod_cameraCalib,
)


class RCNN_only(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, opt, logger, printer, rank=-1):
        super().__init__()

        self.opt = opt
        self.cfg = cfg
        self.if_print = self.opt.debug
        self.logger = logger
        self.printer = printer
        self.rank = rank

        self.cls_names = ["horizon", "pitch", "roll", "vfov", "camH"]

        torch.manual_seed(12344)
        self.RCNN = GeneralizedRCNNRuiMod_cameraCalib(
            cfg,
            opt,
            modules_not_build=["roi_h_heads"],
            logger=self.logger,
            rank=self.rank,
        )

    def init_restore(self, old=False, if_print=False):
        save_dir = self.cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(
            self.opt,
            self.RCNN,
            checkpoint_all_dir=self.opt.checkpoints_folder,
            save_dir=save_dir,
            logger=self.logger,
            if_print=self.if_print,
        )
        # Load backbone
        if "SUN360RCNN" in self.cfg.MODEL.RCNN_WEIGHT_BACKBONE:
            _ = checkpointer.load(
                task_name=self.cfg.MODEL.RCNN_WEIGHT_BACKBONE,
                only_load_kws=["backbone"],
            )
        else:
            _ = checkpointer.load(
                f=self.cfg.MODEL.RCNN_WEIGHT_BACKBONE,
                only_load_kws=["backbone"],
            )

        # Load camera classifiers except camH
        if "SUN360RCNN" in self.cfg.MODEL.RCNN_WEIGHT_CLS_HEAD:
            _ = checkpointer.load(
                task_name=self.cfg.MODEL.RCNN_WEIGHT_CLS_HEAD,
                only_load_kws=["classifier_heads"],
                skip_kws=["camH"],
            )
        else:
            skip_kws_CLS_HEAD = [
                "classifier_%s.predictor" % cls_name for cls_name in self.cls_names
            ]
            replace_kws_CLS_HEAD = [
                "classifier_heads.classifier_%s" % cls_name
                for cls_name in self.cls_names
            ]
            replace_with_kws_CLS_HEAD = ["roi_heads.box"] * len(self.cls_names)
            _ = checkpointer.load(
                f=self.cfg.MODEL.RCNN_WEIGHT_CLS_HEAD,
                only_load_kws=replace_kws_CLS_HEAD,
                skip_kws=skip_kws_CLS_HEAD,
                replace_kws=replace_kws_CLS_HEAD,
                replace_with_kws=replace_with_kws_CLS_HEAD,
            )

    def turn_off_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info(colored("only_enable_camH_bboxPredictor", "white", "on_red"))

    def turn_on_names(self, in_names):
        for name, param in self.named_parameters():
            for in_name in in_names:
                # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = True
                    self.logger.info(
                        colored("turn_ON_in_names: " + in_name, "white", "on_red"),
                    )

    def turn_off_names(self, in_names):
        for name, param in self.named_parameters():
            for in_name in in_names:
                # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = False
                    self.logger.info(
                        colored("turn_False_in_names: " + in_name, "white", "on_red"),
                    )

    def print_net(self):
        count_grads = 0
        for name, param in self.named_parameters():
            print(
                name,
                param.shape,
                white_blue("True") if param.requires_grad else green("False"),
            )
            if param.requires_grad:
                count_grads += 1
        self.logger.info(
            white_blue(
                "---> ALL %d params; %d trainable"
                % (len(list(self.named_parameters())), count_grads),
            ),
        )
        return count_grads

    def forward(
        self,
        # For compatibility with the new model
        input_dict_misc=None,
        image_batch_list=None,
        list_of_bbox_list_cpu=None,
        list_of_oneLargeBbox_list=None,
        im_filename=None,
    ):
        """
        :param images224: torch.Size([8, 3, 224, 224])
        :param image_batch_list: List(np.array)
        :return:
        """
        if im_filename is not None and self.if_print:
            print("in model: im_filename", colored(im_filename, "white", "on_red"))

        if image_batch_list is not None:
            output_RCNN = self.RCNN(
                image_batch_list,
                list_of_bbox_list_cpu,
                list_of_oneLargeBbox_list,
            )
            return output_RCNN
        else:
            return None

    def turn_off_print(self):
        self.if_print = False

    def turn_on_print(self):
        self.if_print = True


if __name__ == "__main__":
    pass
