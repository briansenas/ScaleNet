import itertools
from statistics import mean

import numpy as np
import skimage.io as io
import torch
import torch.distributed as dist
import utils.vis_utils as vis_utils
from maskrcnn_rui.engine.inference import _accumulate_predictions_from_multiple_gpus
from maskrcnn_rui.engine.inference import _dict_to_list
from tqdm import tqdm
from train_batch_combine_RCNNOnly_v5_pose_multiCat import train_batch_combine
from utils.train_utils import reduce_loss_dict
from utils.train_utils import sum_bbox_ratios
from utils.utils_misc import batch_dict_to_list_of_dicts
from utils.utils_misc import colored

np.set_printoptions(precision=3, suppress=True)


def eval_epoch_combine_RCNNOnly(
    model,
    eval_loader,
    device,
    opt,
    bins,
    tid,
    savepath_coco="",
    savepath_cocoPose="",
    writer=None,
    logger=None,
    if_debug=False,
    max_iter=-1,
    if_vis=False,
    if_loss=True,
    prepostfix="",
    epoch=None,
):
    loss_func_l1 = torch.nn.L1Loss()
    if opt.distributed:
        rank = dist.get_rank()
    else:
        rank = 0

    eval_loss_vt_list = []
    eval_loss_vt_layers_dict_list = []

    eval_loss_person_list = []
    vt_loss_allBoxes_dict = {}
    vt_error_fit_allBoxes_dict = {}
    im_filename_list = []

    loss_list = []

    camH_est_list = []
    camH_fit_list = []
    person_hs_list = []
    person_hs_list_layers = [[] for i in range(opt.num_layers)]
    labels_list_all = []

    eval_loss_kp_list = []
    eval_loss_bbox_cls_list = []
    eval_loss_bbox_reg_list = []

    return_dict_epoch = {}

    if_vis_kps = True

    with torch.no_grad():
        with tqdm(eval_loader, desc=f"Ep.{epoch} Eval", disable=rank != 0) as t:
            for i, (
                _,
                inputCOCO_Image_maskrcnnTransform_list,
                W_batch_array,
                H_batch_array,
                yc_batch,
                bboxes_batch_array,
                bboxes_length_batch_array,
                v0_batch,
                f_pixels_yannick_batch,
                im_filename,
                im_file,
                target_maskrcnnTransform_list,
                labels_list,
            ) in enumerate(t):
                # if if_debug:
                #     print("[eval_epoch] i, rank, im_filename", i, rank, im_filename)

                input_dict = {
                    "inputCOCO_Image_maskrcnnTransform_list": inputCOCO_Image_maskrcnnTransform_list,
                    "W_batch_array": W_batch_array,
                    "H_batch_array": H_batch_array,
                    "yc_batch": yc_batch,
                    "bboxes_batch_array": bboxes_batch_array,
                    "bboxes_length_batch_array": bboxes_length_batch_array,
                    "v0_batch": v0_batch,
                    "f_pixels_yannick_batch": f_pixels_yannick_batch,
                    "im_filename": im_filename,
                    "im_file": im_file,
                    "bins": bins,
                    "labels_list": labels_list,
                    "target_maskrcnnTransform_list": target_maskrcnnTransform_list,
                }

                loss_dict, return_dict = train_batch_combine(
                    input_dict,
                    model,
                    device,
                    opt,
                    is_training=False,
                    tid=i,
                    loss_func=loss_func_l1,
                    rank=rank,
                    if_SUN360=False,
                    if_vis=if_vis,
                )
                toreport = {}
                if if_loss:
                    loss_dict_reduced = reduce_loss_dict(
                        loss_dict,
                        mark=i,
                        logger=logger,
                    )

                    if opt.train_cameraCls and opt.train_roi_h and opt.pointnet_camH:
                        eval_loss_vt_list.append(loss_dict_reduced["loss_vt"].item())
                        toreport["loss_vt"] = eval_loss_vt_list[-1]
                        if opt.pointnet_camH_refine:
                            loss_vt_layers_dict_reduced = reduce_loss_dict(
                                return_dict["loss_vt_layers_dict"],
                                mark=i,
                                logger=logger,
                            )  # **average** over multi GPUs
                            eval_loss_vt_layers_dict_list.append(
                                loss_vt_layers_dict_reduced,
                            )

                        # eval_loss_list.append(loss_dict_reduced['loss_vt'].item())
                        if not opt.not_rcnn:
                            eval_loss_person_list.append(
                                loss_dict_reduced["loss_person"].item(),
                            )
                            toreport["loss_person"] = eval_loss_person_list[-1]

                        vt_loss_allBoxes_dict.update(
                            return_dict["vt_loss_allBoxes_dict"],
                        )
                        if opt.fit_derek:
                            vt_error_fit_allBoxes_dict.update(
                                return_dict["vt_error_fit_allBoxes_dict"],
                            )
                            camH_fit_list.append(return_dict["yc_fit_batch"])

                        camH_est_list.append(
                            return_dict["yc_est_batch"].detach().cpu().numpy(),
                        )

                        if not opt.not_rcnn:
                            person_hs_list.append(
                                return_dict["all_person_hs"].detach().cpu().numpy(),
                            )
                            for layer_idx, all_person_hs_layer in enumerate(
                                return_dict["all_person_hs_layers"],
                            ):
                                person_hs_list_layers[layer_idx].append(
                                    all_person_hs_layer.detach().cpu().numpy(),
                                )
                            labels_list_all.append(
                                list(itertools.chain(*input_dict["labels_list"])),
                            )

                    if opt.est_kps:
                        eval_loss_kp_list.append(loss_dict_reduced["loss_kp"].item())
                        toreport["loss_kp"] = eval_loss_kp_list[-1]
                    if opt.est_bbox:
                        eval_loss_bbox_cls_list.append(
                            loss_dict_reduced["loss_bbox_cls"].item(),
                        )
                        eval_loss_bbox_reg_list.append(
                            loss_dict_reduced["loss_bbox_reg"].item(),
                        )
                        toreport["loss_bbox_cls"] = eval_loss_bbox_cls_list[-1]
                        toreport["loss_bbox_reg"] = eval_loss_bbox_cls_list[-1]

                    im_filename_list += list(im_filename)

                    losses = sum(loss for loss in loss_dict_reduced.values())
                    loss_list.append(losses.item())
                    # Visualization cues
                    t.set_postfix(**toreport)
                    t.update()

                # ===== Some vis
                if rank == 0 and if_vis:
                    input_dict_show = return_dict["input_dict_show"]
                    if input_dict_show["num_samples"] > 0:
                        input_dict_show_list = batch_dict_to_list_of_dicts(
                            input_dict_show,
                        )
                        input_dict_show = input_dict_show_list[0]

                    prefix, postfix = prepostfix.split("|")
                    if opt.train_cameraCls:
                        vis_utils.show_cam_bbox(
                            io.imread(input_dict_show["im_path"]),
                            input_dict_show,
                            if_show=False,
                            save_path=savepath_coco,
                            save_name=prefix + "tid%d-iter%d" % (tid, i) + postfix,
                            if_not_detail=True,
                            idx_sample=i,
                        )
                    if opt.est_kps and if_vis_kps:
                        vis_utils.show_box_kps(
                            opt,
                            model,
                            io.imread(input_dict_show["im_path"]),
                            input_dict_show,
                            if_show=False,
                            save_path=savepath_cocoPose,
                            save_name=prefix + "tid%d-iter%d" % (tid, i) + postfix,
                            idx_sample=i,
                        )

                if max_iter != -1 and i > max_iter:
                    break
    # synchronize()

    if if_loss:
        im_filename_list = _accumulate_predictions_from_multiple_gpus(
            im_filename_list,
            only_gather=True,
        )
        return_dict_epoch.update({"im_filename_list": im_filename_list})

        if opt.train_cameraCls and opt.train_roi_h and opt.pointnet_camH:
            vt_loss_allBoxes_dict = _accumulate_predictions_from_multiple_gpus(
                vt_loss_allBoxes_dict,
                return_dict=True,
            )
            # if opt.debug and vt_loss_allBoxes_dict is not None:
            #     print(
            #         "++++",
            #         len(list(vt_loss_allBoxes_dict.keys())),
            #         list(sorted(vt_loss_allBoxes_dict.keys())),
            #     )
            vt_loss_allBoxes_list = _dict_to_list(vt_loss_allBoxes_dict)

            if opt.fit_derek:
                vt_error_fit_allBoxes_dict = _accumulate_predictions_from_multiple_gpus(
                    vt_error_fit_allBoxes_dict,
                    return_dict=True,
                )
                vt_error_fit_allBoxes_list = _dict_to_list(vt_error_fit_allBoxes_dict)
                return_dict_epoch.update(
                    {
                        "eval_loss_vt_list": eval_loss_vt_list,
                        "vt_loss_allBoxes_list": vt_loss_allBoxes_list,
                    },
                )

            return_dict_epoch.update(
                {
                    "eval_loss_vt_list": eval_loss_vt_list,
                    "vt_loss_allBoxes_list": vt_loss_allBoxes_list,
                },
            )
            if not opt.not_rcnn:
                return_dict_epoch.update(
                    {"eval_loss_person_list": eval_loss_person_list},
                )

            camH_est_list_concat = np.concatenate(camH_est_list)  # [N, ]
            camH_est_list_concat_list = _accumulate_predictions_from_multiple_gpus(
                camH_est_list_concat,
                only_gather=True,
            )
            camH_est_list_concat_all = np.concatenate(camH_est_list_concat_list)
            return_dict_epoch.update(
                {"camH_est_list_concat_all": camH_est_list_concat_all},
            )

            if opt.fit_derek:
                camH_fit_list_concat = np.concatenate(camH_fit_list)  # [N, ]
                camH_fit_list_concat_list = _accumulate_predictions_from_multiple_gpus(
                    camH_fit_list_concat,
                    only_gather=True,
                )
                camH_fit_list_concat_all = np.concatenate(camH_fit_list_concat_list)
                return_dict_epoch.update(
                    {"camH_fit_list_concat_all": camH_fit_list_concat_all},
                )

            if not opt.not_rcnn:
                person_hs_list_concat = np.concatenate(person_hs_list)  # [N, ]
                person_hs_list_concat_list = _accumulate_predictions_from_multiple_gpus(
                    person_hs_list_concat,
                    only_gather=True,
                )
                person_hs_list_concat_all = np.concatenate(person_hs_list_concat_list)
                return_dict_epoch.update(
                    {"person_hs_list_concat_all": person_hs_list_concat_all},
                )

                person_hs_list_concat_layers = [
                    np.concatenate(person_hs_list_layer)
                    for person_hs_list_layer in person_hs_list_layers
                ]  # [N, ]
                person_hs_list_concat_list_layers = [
                    _accumulate_predictions_from_multiple_gpus(
                        person_hs_list_concat,
                        only_gather=True,
                    )
                    for person_hs_list_concat in person_hs_list_concat_layers
                ]
                person_hs_list_concat_all_layers = [
                    np.concatenate(person_hs_list_concat_list_layer)
                    for person_hs_list_concat_list_layer in person_hs_list_concat_list_layers
                ]
                return_dict_epoch.update(
                    {
                        "person_hs_list_concat_all_layers": person_hs_list_concat_all_layers,
                    },
                )

                labels_list_all_concat = np.concatenate(labels_list_all)  # [N, ]
                labels_list_all_concat_list = (
                    _accumulate_predictions_from_multiple_gpus(
                        labels_list_all_concat,
                        only_gather=True,
                    )
                )
                labels_list_all_concat_all = np.concatenate(labels_list_all_concat_list)
                return_dict_epoch.update(
                    {"labels_list_all_concat_all": labels_list_all_concat_all},
                )

            if rank == 0 and writer is not None:
                logger.info(
                    colored(
                        "Writing val summaries for epoch %d of task %s..."
                        % (epoch, opt.task_name),
                        "white",
                        "on_blue",
                    ),
                )
                eval_loss_vt_mean = mean(eval_loss_vt_list)
                if opt.pointnet_camH_refine:
                    writer.add_scalar(
                        "loss_eval/eval_vt_loss_meanLayers",
                        eval_loss_vt_mean,
                        tid,
                    )
                    len_layers = len(return_dict["loss_vt_layers_dict"].keys())
                    for loss_idx in range(len_layers):
                        loss_vt_layer_name = "loss_vt_layer_%d" % (
                            loss_idx - len_layers
                        )
                        writer.add_scalar(
                            "loss_eval/" + loss_vt_layer_name,
                            mean(
                                [
                                    eval_loss_vt_layers_dict[loss_vt_layer_name].item()
                                    for eval_loss_vt_layers_dict in eval_loss_vt_layers_dict_list
                                ],
                            ),
                            tid,
                        )
                    last_layer_loss = mean(
                        [
                            eval_loss_vt_layers_dict["loss_vt_layer_-1"].item()
                            for eval_loss_vt_layers_dict in eval_loss_vt_layers_dict_list
                        ],
                    )
                    writer.add_scalar("loss_eval/eval_vt_loss", last_layer_loss, tid)
                    return_dict_epoch.update({"eval_loss_vt": last_layer_loss})
                else:
                    writer.add_scalar("loss_eval/eval_vt_loss", eval_loss_vt_mean, tid)
                    return_dict_epoch.update({"eval_loss_vt": eval_loss_vt_mean})
                if not opt.not_rcnn:
                    writer.add_scalar(
                        "loss_eval/eval_person_loss",
                        mean(eval_loss_person_list),
                        tid,
                    )
                    writer.add_scalar(
                        "loss_eval/eval_person_loss_div_w",
                        mean(eval_loss_person_list) / opt.cfg.SOLVER.PERSON_WEIGHT,
                        tid,
                    )
                    person_hs = [
                        person_h
                        for person_h, label_h in zip(
                            person_hs_list_concat_all,
                            labels_list_all_concat_all,
                        )
                        if label_h == "person"
                    ]
                    if person_hs:
                        writer.add_histogram(
                            "val/person_hs_est_all",
                            person_hs,
                            tid,
                            bins="doane",
                        )
                    car_hs = [
                        person_h
                        for person_h, label_h in zip(
                            person_hs_list_concat_all,
                            labels_list_all_concat_all,
                        )
                        if label_h == "car"
                    ]
                    if car_hs:
                        writer.add_histogram(
                            "val/car_hs_est_all",
                            car_hs,
                            tid,
                            bins="doane",
                        )

                    for layer_idx, person_hs_list_concat_all_layer in enumerate(
                        person_hs_list_concat_all_layers,
                    ):
                        person_hs = [
                            person_h
                            for person_h, label_h in zip(
                                person_hs_list_concat_all_layer,
                                labels_list_all_concat_all,
                            )
                            if label_h == "person"
                        ]
                        if person_hs:
                            writer.add_histogram(
                                "val_layers/person_hs_est_all_layer%d" % layer_idx,
                                person_hs,
                                tid,
                                bins="doane",
                            )
                        car_hs = [
                            person_h
                            for person_h, label_h in zip(
                                person_hs_list_concat_all_layer,
                                labels_list_all_concat_all,
                            )
                            if label_h == "car"
                        ]
                        if car_hs:
                            writer.add_histogram(
                                "val_layers/car_hs_est_all_layer%d" % layer_idx,
                                car_hs,
                                tid,
                                bins="doane",
                            )

                # NOTE: only call this function from rank = 0 due to writer.
                _, thres_ratio_dict = sum_bbox_ratios(
                    writer,
                    vt_loss_allBoxes_list,
                    tid,
                    prefix="eval",
                    title="vt_loss_allBoxes",
                )
                if opt.fit_derek:
                    sum_bbox_ratios(
                        writer,
                        vt_error_fit_allBoxes_list,
                        tid,
                        prefix="eval-fit",
                        title="vt_error_fit_allBoxes",
                    )
                    writer.add_histogram(
                        "val/camH_fit_all",
                        camH_fit_list_concat_all,
                        tid,
                        bins="doane",
                    )
                return_dict_epoch.update({"thres_ratio_dict": thres_ratio_dict})
                writer.add_histogram(
                    "val/camH_est_all",
                    camH_est_list_concat_all,
                    tid,
                    bins="doane",
                )

        if opt.est_kps:
            return_dict_epoch.update({"eval_loss_kp_list": eval_loss_kp_list})
            if rank == 0 and writer is not None:
                eval_loss_kp_mean = mean(eval_loss_kp_list)
                writer.add_scalar(
                    "loss_eval/eval_kp_loss",
                    eval_loss_kp_mean / (opt.weight_kps + 1e-8),
                    tid,
                )
                return_dict_epoch.update({"eval_loss_kp": eval_loss_kp_mean})
        if opt.est_bbox:
            return_dict_epoch.update(
                {"eval_loss_bbox_cls_list": eval_loss_bbox_cls_list},
            )
            return_dict_epoch.update(
                {"eval_loss_bbox_reg_list": eval_loss_bbox_reg_list},
            )
            if rank == 0 and writer is not None:
                eval_loss_bbox_cls_mean = mean(eval_loss_bbox_cls_list)
                writer.add_scalar(
                    "loss_eval/eval_bbox_cls_loss",
                    eval_loss_bbox_cls_mean / (opt.weight_kps + 1e-8),
                    tid,
                )
                return_dict_epoch.update(
                    {"eval_loss_bbox_cls": eval_loss_bbox_cls_mean},
                )
                eval_loss_bbox_reg_mean = mean(eval_loss_bbox_reg_list)
                writer.add_scalar(
                    "loss_eval/eval_bbox_reg_loss",
                    eval_loss_bbox_reg_mean / (opt.weight_kps + 1e-8),
                    tid,
                )
                return_dict_epoch.update(
                    {"eval_loss_bbox_reg": eval_loss_bbox_reg_mean},
                )

        eval_loss_sum = np.sum(np.asarray(loss_list))
        if rank == 0 and writer is not None:
            writer.add_scalar("loss_eval/eval_loss_sum_coco", eval_loss_sum, tid)
        return_dict_epoch.update({"eval_loss_sum_coco": eval_loss_sum})

        if rank == 0 and epoch is not None:
            writer.add_scalar("training/eval_epoch", epoch, tid)

    return return_dict_epoch
