import logging
import ntpath
import os
import pathlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dataset_cvpr import bins2horizon
from dataset_cvpr import bins2pitch
from dataset_cvpr import bins2roll
from dataset_cvpr import bins2vfov
from imageio import imread
from imageio import imsave
from matplotlib.patches import Rectangle
from panorama_cropping_dataset_generation.debugging import showHorizonLine
from panorama_cropping_dataset_generation.debugging import showHorizonLineFromHorizon
from PIL import Image
from scipy.special import softmax
from utils.utils_misc import colored
from utils.utils_misc import green

PATH = pathlib.Path(__file__).parent.resolve()


def show_cam_bbox(
    img,
    input_dict_show,
    save_path=".",
    save_name="tmp",
    if_show=False,
    if_save=True,
    figzoom=1.0,
    if_pause=True,
    if_return=False,
    if_not_detail=False,
    idx_sample=0,
):
    input_dict = {
        "tid": -1,
        "yc_fit": -1,
        "yc_est": -1,
        "f_est_px": -1,
        "f_est_mm": -1,
        "pitch_est_angle": -1,
    }
    # SHOW IMAGE, HORIZON FROM YANNICK
    input_dict.update(input_dict_show)
    # Turn interactive plotting off
    if if_show == False:
        plt.ioff()
    fig = plt.figure(figsize=(10 * figzoom, 10 * figzoom))
    # plt.subplot(4, 1, [1, 2, 3])
    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0:3, :])
    plt.imshow(img)
    if "yc_est_list" in input_dict:
        add_yc_list_str = "(%s)" % (
            ", ".join(["%.2f" % yc_est for yc_est in input_dict["yc_est_list"]])
        )
    else:
        add_yc_list_str = ""

    if "f_pixels_est_mm_list" in input_dict:
        add_fmm_list_str = "(%s)" % (
            ", ".join(
                ["%.2f" % fmm_est for fmm_est in input_dict["f_pixels_est_mm_list"]],
            )
        )
    else:
        add_fmm_list_str = ""

    if "v0_est_list" in input_dict:
        add_v0_list_str = "(%s)" % (
            ", ".join(["%.2f" % v0_est for v0_est in input_dict["v0_est_list"]])
        )
        add_v0_list_str = "v0=%.2f" % sum(input_dict["v0_est_list"]) + add_v0_list_str
    else:
        add_v0_list_str = ""

    plt.title(
        "[%s-%d-%s] H_camFit=%.2f, H_camEst=%.2f %s; f_est=%.2f mm %s; pitch=%.2f degree; %s"
        % (
            input_dict["task_name"].split("_")[0],
            input_dict["tid"],
            input_dict["im_filename"][-6:],
            input_dict["yc_fit"],
            input_dict["yc_est"],
            add_yc_list_str,
            input_dict["f_est_mm"],
            add_fmm_list_str,
            input_dict["pitch_est_angle"],
            add_v0_list_str,
        ),
        fontsize="small",
    )

    W = input_dict["W"]
    H = input_dict["H"]

    if "v0_cocoPredict" in input_dict:
        v0_cocoPredict = input_dict["v0_cocoPredict"]  # [top H, bottom 0],
        plt.plot(
            [0.0, W / 2.0],
            [H - v0_cocoPredict, H - v0_cocoPredict],
            "w--",
            "linewidth",
            50,
        )

    # SHOW HORIZON EST
    if "v0_batch_predict" in input_dict:
        v0_batch_predict = input_dict[
            "v0_batch_predict"
        ]  # (H = top of the image, 0 = bottom of the image)
        plt.plot(
            [0.0, W - 1.0],
            [H - v0_batch_predict, H - v0_batch_predict],
            linestyle="-",
            linewidth=2,
            color="black",
        )
        if not if_not_detail:
            plt.text(
                W / 4.0,
                H - v0_batch_predict,
                "v0_predict %.2f" % (1.0 - v0_batch_predict / H),
                fontsize=8,
                weight="bold",
                color="black",
                bbox=dict(facecolor="w", alpha=0.5, boxstyle="round", edgecolor="none"),
            )

    if "v0_batch_est" in input_dict:
        v0_batch_est = input_dict[
            "v0_batch_est"
        ]  # (H = top of the image, 0 = bottom of the image)
        plt.plot(
            [0.0, W - 1.0],
            [H - v0_batch_est, H - v0_batch_est],
            linestyle="-",
            linewidth=2,
            color="lime",
        )
        if not if_not_detail:
            plt.text(
                W / 2.0,
                H - v0_batch_est,
                "v0_est %.2f" % (1.0 - v0_batch_est / H),
                fontsize=8,
                weight="bold",
                color="lime",
                bbox=dict(facecolor="w", alpha=0.5, boxstyle="round", edgecolor="none"),
            )

    if "v0_batch_est_0" in input_dict:
        v0_batch_est = input_dict[
            "v0_batch_est_0"
        ]  # (H = top of the image, 0 = bottom of the image)
        plt.plot(
            [0.0, W - 1.0],
            [H - v0_batch_est, H - v0_batch_est],
            linestyle="-",
            linewidth=2,
            color="aquamarine",
        )
        if not if_not_detail:
            plt.text(
                0.0,
                H - v0_batch_est,
                "v0_est_0 %.2f" % (1.0 - v0_batch_est / H),
                fontsize=8,
                weight="bold",
                color="aquamarine",
                bbox=dict(facecolor="w", alpha=0.5, boxstyle="round", edgecolor="none"),
            )

    ax = plt.gca()

    if "bbox_gt" in input_dict:
        for bbox in input_dict["bbox_gt"]:
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=5,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

    if "bbox_est" in input_dict:
        for bbox in input_dict["bbox_est"]:
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor="b",
                facecolor="none",
            )
            ax.add_patch(rect)

    if "bbox_fit" in input_dict:
        for bbox in input_dict["bbox_fit"]:
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

    if "bbox_h" in input_dict:
        if "vt_camEst_N_delta_est_list" in input_dict:
            vt_camEst_N_delta_list = input_dict["vt_camEst_N_delta_est_list"]
            vt_camEst_N_delta_list_sample = []
            for sample_idx in range(len(input_dict["bbox_h"])):
                vt_camEst_N_delta_list_sample.append(
                    [
                        vt_camEst_N_delta_layer[sample_idx]
                        for vt_camEst_N_delta_layer in vt_camEst_N_delta_list
                    ],
                )
            if not if_not_detail:
                for bbox, vt_camEst_person_delta_person_layers in zip(
                    input_dict["bbox_gt"],
                    vt_camEst_N_delta_list_sample,
                ):
                    add_vt_camEst_person_delta_person_list_str = "(%s)" % (
                        ", ".join(
                            [
                                "%.2f" % vt_camEst_person_delta_person
                                for vt_camEst_person_delta_person in vt_camEst_person_delta_person_layers
                            ],
                        )
                    )
                    plt.text(
                        bbox[0],
                        bbox[1] + bbox[3],
                        "Err %s" % (add_vt_camEst_person_delta_person_list_str),
                        fontsize=7,
                        bbox=dict(facecolor="aquamarine", alpha=0.5),
                    )

        if "person_hs_est_list" not in input_dict:
            for y_person, bbox in zip(input_dict["bbox_h"], input_dict["bbox_gt"]):
                plt.text(
                    bbox[0],
                    bbox[1],
                    "%.2f" % (y_person),
                    fontsize=12,
                    weight="bold",
                    bbox=dict(facecolor="white", alpha=0.5),
                )
        else:
            person_hs_est_list = input_dict["person_hs_est_list"]
            person_hs_est_list_sample = []
            for sample_idx in range(len(input_dict["bbox_h"])):
                person_hs_est_list_sample.append(
                    [
                        person_hs_est_layer[sample_idx]
                        for person_hs_est_layer in person_hs_est_list
                    ],
                )
            for y_person, bbox, y_person_layers in zip(
                input_dict["bbox_h"],
                input_dict["bbox_gt"],
                person_hs_est_list_sample,
            ):
                if not if_not_detail:
                    add_y_person_list_str = "(%s)" % (
                        ", ".join(["%.2f" % y_person for y_person in y_person_layers])
                    )
                else:
                    add_y_person_list_str = ""
                plt.text(
                    bbox[0],
                    bbox[1],
                    f"{y_person:.2f} {add_y_person_list_str}",
                    fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.5),
                )
            if "bbox_h_canonical" in input_dict:
                for y_person, y_person_canonical, bbox, y_person_layers in zip(
                    input_dict["bbox_h"],
                    input_dict["bbox_h_canonical"],
                    input_dict["bbox_gt"],
                    person_hs_est_list_sample,
                ):
                    plt.text(
                        bbox[0],
                        bbox[1] + 15,
                        "%.2f C" % (y_person_canonical),
                        fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.5),
                    )

    if "bbox_loss" in input_dict and not if_not_detail:
        for vt_loss, bbox in zip(input_dict["bbox_loss"], input_dict["bbox_est"]):
            plt.text(
                bbox[0] + bbox[2] - 8,
                bbox[1] + bbox[3] / 2.0 * 1.5 - 8,
                "%.2f" % (vt_loss),
                fontsize=8,
                weight="bold",
                color="white",
                bbox=dict(facecolor="b", alpha=0.5),
            )

    if "v0_batch_from_pitch_vfov" in input_dict:
        v0_batch_from_pitch_vfov = input_dict_show["v0_batch_from_pitch_vfov"]
        plt.plot(
            [0.0, W],
            [H - v0_batch_from_pitch_vfov, H - v0_batch_from_pitch_vfov],
            linestyle="-.",
            linewidth=2,
            color="blue",
        )
        plt.text(
            W / 4.0 * 3.0,
            H - v0_batch_from_pitch_vfov,
            "v0_from_pitch_vfov %.2f" % (1.0 - v0_batch_from_pitch_vfov / H),
            fontsize=8,
            weight="bold",
            color="blue",
            bbox=dict(facecolor="w", alpha=0.5, boxstyle="round", edgecolor="none"),
        )

    plt.xlim([0, W])
    plt.ylim([H, 0])

    # SHOW HORIZON ARGMAX
    if "output_horizon_COCO" in input_dict and not if_not_detail:
        output_horizon = input_dict["output_horizon_COCO"]
        horizon_bins = input_dict["horizon_bins"]
        ax2 = fig.add_subplot(gs[3, 0])
        vis_output_softmax_argmax(
            output_horizon,
            horizon_bins,
            ax2,
            title="horizon-" + input_dict["reduce_method"],
        )

    if "output_camH_COCO" in input_dict and not if_not_detail:
        output_camH = input_dict["output_camH_COCO"]
        camH_bins = input_dict["camH_bins"]
        ax3 = fig.add_subplot(gs[3, 1])
        vis_output_softmax_argmax(
            output_camH,
            camH_bins,
            ax3,
            title="camH-" + input_dict["reduce_method"],
        )

    if if_show and not if_return:
        plt.show()
        if if_pause:
            if_delete = input(colored("Pause", "white", "on_blue"))

    if if_save:
        vis_path = os.path.join(save_path, save_name + ".jpg")
        fig.savefig(vis_path)
        if idx_sample == 0:
            print("Vis saved to " + vis_path)

    if if_return:
        return fig, ax1
    else:
        plt.close(fig)


def vis_output_softmax_argmax(output_camH, camH_bins, ax, title=""):
    camH_softmax_prob = softmax(output_camH, axis=0)
    camH_max_prob = np.max(camH_softmax_prob)
    plt.plot(camH_bins, camH_softmax_prob)
    camH_softmax_estim = np.sum(camH_softmax_prob * camH_bins)
    plt.plot([camH_softmax_estim, camH_softmax_estim], [0, camH_max_prob * 1.05], "--")
    camH_argmax = camH_bins[np.argmax(output_camH)]
    plt.plot([camH_argmax, camH_argmax], [0, camH_max_prob * 1.05])
    plt.grid()
    ax.set_title(
        "{}: softmax {:.2f}, argmax {:.2f}".format(
            title,
            camH_softmax_estim,
            camH_argmax,
        ),
        fontsize="small",
    )


def show_box_kps(
    opt,
    model,
    img,
    input_dict_show,
    save_path=".",
    save_name="tmp",
    if_show=False,
    if_save=True,
    figzoom=1.0,
    if_pause=True,
    if_return=False,
    idx_sample=0,
    select_top=True,
    predictions_override=None,
):
    image_sizes_ori = [
        (input_dict_show["W_batch_array"], input_dict_show["H_batch_array"]),
    ]
    if predictions_override is None:
        if (
            "predictions" not in input_dict_show
            or input_dict_show["predictions"] is None
        ):
            return
        predictions = [input_dict_show["predictions"]]
    else:
        predictions = predictions_override
    if opt.distributed:
        _, prediction_list_ori = model.module.RCNN.post_process(
            predictions,
            image_sizes_ori,
        )
        image_batch_list_ori = [img]
        result_list, top_prediction_list = model.module.RCNN.select_and_vis_bbox(
            prediction_list_ori,
            image_batch_list_ori,
        )
    else:
        _, prediction_list_ori = model.RCNN.post_process(
            predictions,
            image_sizes_ori,
        )
        image_batch_list_ori = [img]
        result_list, top_prediction_list = model.RCNN.select_and_vis_bbox(
            prediction_list_ori,
            image_batch_list_ori,
            select_top=select_top,
        )
    input_dict_show["result_list_pose"] = result_list
    target_list = [input_dict_show["target_maskrcnnTransform_list"]]
    for _, (_, result) in enumerate(zip(target_list, result_list)):
        # bboxes_gt = target.get_field('boxlist_ori').convert("xywh").bbox.numpy()
        if if_show == False:
            plt.ioff()
        fig = plt.figure(figsize=(10 * figzoom, 10 * figzoom))
        plt.imshow(result)
        plt.title(
            "[%d-%s]" % (input_dict_show["tid"], input_dict_show["im_filename"][-6:]),
        )
        if if_show and not if_return:
            plt.show()
            if if_pause:
                _ = input(colored("Pause", "white", "on_blue"))
        if if_save:
            vis_path = os.path.join(save_path, save_name + ".jpg")
            fig.savefig(vis_path)
            if idx_sample == 0:
                print("Vis saved to " + vis_path)
        plt.close(fig)
    return result_list, top_prediction_list


# def vis_pose(tid, save_path, im_paths)
def vis_SUN360(
    tid,
    save_path,
    im_paths,
    output_horizon,
    output_pitch,
    output_roll,
    output_vfov,
    horizon_num,
    pitch_num,
    roll_num,
    vfov_num,
    f_num,
    sensor_size_num,
    rank,
    if_vis=True,
    if_save=False,
    min_samples=5,
    logger=None,
    prepostfix="",
    idx_sample=0,
):
    if logger is None:
        logger = logging.getLogger("vis_SUN360")

    if rank == 0 and if_vis:
        logger.info(green("Visualizing SUN360..... potentially save to" + save_path))
    im2_list = []
    horizon_list = []
    pitch_list = []
    roll_list = []
    vfov_list = []
    f_mm_list = []

    for idx in range(min(min_samples, len(im_paths))):
        im = Image.fromarray(imread(im_paths[idx])[:, :, :3])
        if len(im.getbands()) == 1:
            im = Image.fromarray(np.tile(np.asarray(im)[:, :, np.newaxis], (1, 1, 3)))

        horizon_disc = output_horizon[idx].detach().cpu().numpy().squeeze()
        pitch_disc = output_pitch[idx].detach().cpu().numpy().squeeze()
        roll_disc = output_roll[idx].detach().cpu().numpy().squeeze()
        vfov_disc = output_vfov[idx].detach().cpu().numpy().squeeze()
        vfov_disc[..., 0] = -35
        vfov_disc[..., -1] = -35

        horizon = bins2horizon(horizon_disc)
        pitch = bins2pitch(pitch_disc)
        roll = bins2roll(roll_disc)
        vfov = bins2vfov(vfov_disc)
        h, _ = im.size
        f_pix = h / 2.0 / np.tan(vfov / 2.0)
        sensor_size = sensor_size_num[idx]
        f_mm = f_pix / h * sensor_size

        horizon_list.append(horizon)
        pitch_list.append(pitch)
        roll_list.append(roll)
        vfov_list.append(vfov)
        f_mm_list.append(f_mm)

        if if_vis:
            im2 = np.asarray(im).copy()
            im2, _ = showHorizonLine(
                im2,
                vfov,
                pitch,
                0.0,
                focal_length=f_mm,
                color=(0, 0, 255),
                width=4,
            )  # Blue: horizon converted from camera params WITHOUT roll
            im2 = showHorizonLineFromHorizon(
                im2,
                horizon,
                color=(255, 255, 0),
                width=4,
                debug=True,
            )  # Yellow: est horizon v0

            horizon_gt = horizon_num[idx]
            pitch_gt = pitch_num[idx]
            roll_gt = roll_num[idx]
            vfov_gt = vfov_num[idx]
            f_gt = f_num[idx]
            im2 = showHorizonLineFromHorizon(
                im2,
                horizon_gt,
                color=(255, 255, 255),
                width=3,
                debug=True,
                GT=True,
            )  # White: GT horizon without roll
            im2, _ = showHorizonLine(
                im2,
                vfov,
                pitch,
                roll,
                focal_length=f_mm,
                debug=True,
                color=(0, 0, 255),
                width=2,
            )  # Blue: horizon converted from camera params with roll
            im2, _ = showHorizonLine(
                im2,
                vfov_gt,
                pitch_gt,
                roll_gt,
                focal_length=f_gt,
                debug=True,
                GT=True,
                color=(255, 255, 255),
                width=1,
            )  # White: GT horizon

            if if_save:
                prefix, postfix = prepostfix.split("|")
                im_save_path = os.path.join(
                    save_path,
                    prefix
                    + "tid%d-rank%d-idx%d" % (tid, rank, idx)
                    + postfix
                    + "-"
                    + ntpath.basename(im_paths[idx])
                    + f"-f{f_mm:.2f}-GT{f_gt:.2f}.jpg",
                )
                imsave(im_save_path, im2)
                if idx_sample == 0:
                    print("Vis saved to " + im_save_path)

            im2_list.append(im2)

    # epochs_evaled.append(epoch)
    return_dict = {
        "horizon_list": horizon_list,
        "pitch_list": pitch_list,
        "roll_list": roll_list,
        "vfov_list": vfov_list,
        "f_mm_list": f_mm_list,
    }
    return im2_list, return_dict


def blender_render(
    input_dict_show,
    output_RCNN,
    im_file,
    save_path="rendering/blender",
    if_show=True,
    save_name="",
    tmp_code="iamgroot",
    render_type="cylinder",
    if_compact=False,
    pick=-1,
    grid=False,
    *,
    blender_path="/snap/bin/blender",
    current_dir=".",
):
    assert render_type in ["chair", "cylinder"]
    scene_path = current_dir + "/rendering/scene_chair_fix.blend"
    script_path = current_dir + "/rendering/render_coco_obj_ref.py"

    W = input_dict_show["W_batch_array"]
    H = input_dict_show["H_batch_array"]

    insertion_points_xy_list = []
    bboxes_filter = input_dict_show["bbox_gt"]
    bbox_hs_list = [a.item() for a in input_dict_show["bbox_h"]]
    for bbox in bboxes_filter:
        insertion_points_xy_list.append([bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3]])

    if pick != -1:
        insertion_points_xy_list = [insertion_points_xy_list[pick]]
        bbox_hs_list = [bbox_hs_list[pick]]

    u_start_end = [W / 4.0, W / 4.0 * 3.0]
    u_grid_size = (u_start_end[1] - u_start_end[0]) / 2.0
    v_start_end = [H - input_dict_show["v0_batch_from_pitch_vfov"] + 10, H]
    v_grid_size = (v_start_end[1] - v_start_end[0]) / 4.0

    if grid:
        for u in range(3):
            for v in range(5):
                #         if v > 0:
                #             continue
                insertion_points_xy_list.append(
                    [
                        u_start_end[0] + u_grid_size * u,
                        v_start_end[0] + v_grid_size * v,
                    ],
                )
                bbox_hs_list += [2.5 / (v + 1)]

    tmp_dir = Path("rendering/tmp_dir")
    tmp_dir.mkdir(exist_ok=True)
    npy_path = tmp_dir / ("tmp_insert_pts_" + tmp_code)
    np.save(str(npy_path), insertion_points_xy_list)
    npy_path = tmp_dir / ("tmp_bbox_hs_" + tmp_code)
    np.save(str(npy_path), bbox_hs_list)

    im_filepath = im_file[0]
    insertion_points_x = -1
    insertion_points_y = -1
    ppitch = output_RCNN["pitch_batch_est"].cpu().numpy()[0]
    ffpixels = output_RCNN["f_pixels_batch_est"].cpu().numpy()[0]
    vvfov = output_RCNN["vfov_estim"].cpu().numpy()[0]
    hhfov = np.arctan(W / 2.0 / ffpixels) * 2.0
    h_cam = output_RCNN["yc_est_batch"].cpu().numpy()[0]

    npy_path = Path(current_dir) / tmp_dir
    rendering_command = "{} {} --background --python {}".format(
        blender_path,
        scene_path,
        script_path,
    )
    rendering_command_append = (
        " -- -npy_path %s -img_path %s -tmp_code %s -H %d -W %d -insertion_points_x %d -insertion_points_y %d -pitch %.6f -fov_h %.6f -fov_v %.6f -cam_h %.6f -obj-name %s"
        % (
            str(npy_path),
            im_filepath,
            tmp_code,
            H,
            W,
            insertion_points_x,
            insertion_points_y,
            ppitch,
            hhfov,
            vvfov,
            h_cam,
            render_type,
        )
    )
    rendering_command = rendering_command + rendering_command_append
    print(rendering_command)

    os.system(rendering_command)

    if if_show == False:
        plt.ioff()
    fig = plt.figure(figsize=(15, 15), frameon=False)

    def full_frame(width=None, height=None):
        mpl.rcParams["savefig.pad_inches"] = 0
        figsize = None if width is None else (width, height)
        _ = plt.figure(figsize=figsize)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)

    if if_compact:
        full_frame(15, 15)
    render_file = current_dir + "/rendering/render/render_all_%s.png" % tmp_code
    im_render = plt.imread(render_file)
    plt.imshow(im_render)

    ax = plt.gca()
    input_dict = input_dict_show
    if "bbox_gt" in input_dict:
        for bbox in input_dict["bbox_gt"]:
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

    if "bbox_est" in input_dict:
        for bbox in input_dict["bbox_est"]:
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor="b",
                facecolor="none",
            )
            ax.add_patch(rect)

    v0_batch_from_pitch_vfov = input_dict_show["v0_batch_from_pitch_vfov"]
    plt.plot(
        [0.0, W],
        [H - v0_batch_from_pitch_vfov, H - v0_batch_from_pitch_vfov],
        linestyle="-.",
        linewidth=2,
        color="blue",
    )

    for y_person, bbox in zip(input_dict["bbox_h"], input_dict["bbox_gt"]):
        plt.text(
            bbox[0],
            bbox[1],
            "%.2f" % (y_person),
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    if if_compact:
        ax.set_axis_off()
        plt.axis("off")
    plt.xlim([0, W])
    plt.ylim([H, 0])
    plt.autoscale(tight=True)
    image_dir = os.path.join(current_dir, save_path)
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(
        os.path.join(image_dir, "%s" % (save_name)),
        dpi=100,
        bbox_inches="tight",
    )

    if if_show:
        plt.show()

    plt.close(fig)
