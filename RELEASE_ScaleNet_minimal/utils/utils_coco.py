import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread
from imageio import imsave
from PIL import Image
from PIL import ImageDraw
from scipy.stats import norm

PATH = pathlib.Path(__file__).parent.resolve()


def merge_bboxes(bboxes):
    max_x1y1x2y2 = [np.inf, np.inf, -np.inf, -np.inf]
    for bbox in bboxes:
        max_x1y1x2y2 = [
            min(max_x1y1x2y2[0], bbox[0]),
            min(max_x1y1x2y2[1], bbox[1]),
            max(max_x1y1x2y2[2], bbox[2] + bbox[0]),
            max(max_x1y1x2y2[3], bbox[3] + bbox[1]),
        ]
    return [
        max_x1y1x2y2[0],
        max_x1y1x2y2[1],
        max_x1y1x2y2[2] - max_x1y1x2y2[0],
        max_x1y1x2y2[3] - max_x1y1x2y2[1],
    ]


def check_clear(ann, vis=False, debug=False):
    kps = np.asarray(ann["keypoints"]).reshape(-1, 3)
    if debug:
        print(np.hstack((np.arange(kps.shape[0]).reshape((-1, 1)), kps)))

    if vis:
        pass 

    eyes_ys = kps[1:5, 1]
    eyes_ys_valid_idx = eyes_ys != 0
    eyes_ys_valid = eyes_ys[eyes_ys_valid_idx]
    ankles_ys = kps[15:17, 1]
    ankles_ys_valid_idx = ankles_ys != 0
    ankles_ys_valid = ankles_ys[ankles_ys_valid_idx]
    if eyes_ys_valid.size == 0 or ankles_ys_valid.size == 0:
        return False

    should_min_y_idx = np.argmin(eyes_ys_valid)  # two eyes
    should_max_y_idx = np.argmax(ankles_ys_valid)  # two ankles

    kps_valid = kps[kps[:, 1] != 0, :]

    if debug:
        print(
            eyes_ys_valid[should_min_y_idx],
            np.min(kps_valid[:, 1]),
            kps[15:17, 1][should_max_y_idx],
            np.max(kps_valid[:, 1]),
            kps[1:5, 2],
            kps[15:17, 2],
        )

    return (
        eyes_ys_valid[should_min_y_idx] == np.min(kps_valid[:, 1])
        and ankles_ys_valid[should_max_y_idx] == np.max(kps_valid[:, 1])
        and np.any(np.logical_or(kps[1:5, 2] == 1, kps[1:5, 2] == 2))
        and np.any(np.logical_or(kps[15:17, 2] == 1, kps[15:17, 2] == 2))
    )


def check_valid_surface(cats):
    green_cats_exception = {
        "water": "",
        "ground": "",
        "solid": "",
        "vegetation": ["-", "flower", "tree"],
        "floor": "",
        "plant": ["+", "grass", "leaves"],
    }
    if_green = False
    for super_cat in green_cats_exception.keys():
        if cats[0] == super_cat:
            sub_cats = green_cats_exception[super_cat]
            if sub_cats == "":
                if_green = True
            elif sub_cats[0] == "-":
                if cats[1] not in sub_cats[1:]:
                    if_green = True
            elif sub_cats[0] == "+":
                if cats[1] in sub_cats[1:]:
                    if_green = True
    return if_green


def fpix_to_fmm(f, H, W):
    sensor_diag = 43  # full-frame sensor: 43mm
    img_diag = np.sqrt(H**2 + W**2)
    f_mm = f / img_diag * sensor_diag
    return f_mm


def fpix_to_fmm_croped(f, H, W):
    sensor_size = [24, 36]

    if H / W < 1.0:
        if H / W < sensor_size[0] / sensor_size[1]:
            f_mm = f / W * sensor_size[1]
        else:
            f_mm = f / H * sensor_size[0]
    else:
        if H / W < sensor_size[0] / sensor_size[1]:
            f_mm = f / W * sensor_size[1]
        else:
            f_mm = f / H * sensor_size[0]

    return f_mm


def fmm_to_fpix(f, H, W):
    sensor_diag = np.sqrt(36**2 + 24**2)  # full-frame sensor: 43mm (36mm * 24mm)
    img_diag = np.sqrt(H**2 + W**2)
    f_pix = f / sensor_diag * img_diag
    return f_pix


def fmm_to_fpix_th(f, H, W):
    sensor_diag = np.sqrt(36**2 + 24**2)  # full-frame sensor: 43mm (36mm * 24mm)
    img_diag = torch.sqrt(H**2 + W**2)
    f_pix = f / sensor_diag * img_diag
    return f_pix


def fpix_to_fmm_th(f, H, W):
    sensor_diag = np.sqrt(36**2 + 24**2)  # full-frame sensor: 43mm (36mm * 24mm)
    img_diag = torch.sqrt(H**2 + W**2)
    f_mm = f * sensor_diag / img_diag
    return f_mm


def drawLine(image, hl, hr, leftright=(None, None), color=(0, 255, 0), width=5):
    if np.isnan([hl, hr]).any():
        return image

    h, w, c = image.shape
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype("uint8")

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    l = hl * h
    r = hr * h

    b = 0

    draw.line((b, l, w, r), fill=color, width=width)
    return np.array(im)


def vis_yannick(yannick_results, image_file):
    output_dir = os.path.join(PATH, "tmpdir_adobe")
    os.makedirs(output_dir, exist_ok=True)
    horizon_visible = yannick_results["horizon_visible"][0][0]
    pitch = yannick_results["pitch"][0][0]
    roll = yannick_results["roll"][0][0]
    vfov = yannick_results["vfov"][0][0]
    distortion = yannick_results["distortion"][0][0]
    p_bins = yannick_results["p_bins"][0]
    r_bins = yannick_results["r_bins"][0]

    im = Image.fromarray(imread(image_file))
    if len(im.getbands()) == 1:
        im = Image.fromarray(np.tile(np.asarray(im)[:, :, np.newaxis], (1, 1, 3)))
    imh, imw = im.size[:2]

    plt.figure(figsize=(30, 10))
    if horizon_visible:
        hl, hr = pitch - np.tan(roll) / 2, pitch + np.tan(roll) / 2
        im = drawLine(np.asarray(im), hl, hr)
    im = np.asarray(im)
    imsave(os.path.join(output_dir, image_file), im)

    plt.subplot(131)
    plt.imshow(im)

    # imsave(os.path.join(output_dir, debug_filename), im)

    # plt.clf()
    plt.subplot(132)
    plt.plot(p_bins)
    plt.subplot(133)
    plt.plot(r_bins)
    # plt.savefig(os.path.join(output_dir, os.path.splitext(debug_filename)[0] + "_prob.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    # print("{}: {}, {}, {}, {}, {}".format(debug_filename, horizon_visible, pitch, roll, vfov, distortion))


def getBins(minval, maxval, sigma, alpha, beta, kappa):
    """Remember, bin 0 = below value! last bin mean >= maxval"""
    x = np.linspace(minval, maxval, 255)

    rv = norm(0, sigma)
    pdf = rv.pdf(x)
    pdf /= pdf.max()
    pdf *= alpha
    pdf = pdf.max() * beta - pdf
    cumsum = np.cumsum(pdf)
    cumsum = cumsum / cumsum.max() * kappa
    cumsum -= cumsum[pdf.size // 2]

    return cumsum


def make_bins_layers_list(x_bins_lowHigh_list):
    x_bins_layers_list = []
    for _, x_bins_lowHigh in enumerate(x_bins_lowHigh_list):
        x_bins = np.linspace(x_bins_lowHigh[0], x_bins_lowHigh[1], 255)
        x_bins_centers = x_bins.copy()
        x_bins_centers[:-1] += np.diff(x_bins_centers) / 2
        x_bins_centers = np.append(x_bins_centers, x_bins_centers[-1])  # 42 bins
        x_bins_layers_list.append(x_bins_centers)
    return x_bins_layers_list


bins_lowHigh_list_dict = {}

yc_bins_lowHigh_list = [
    [0.5, 5.0],
    [-0.3, 0.3],
    [-0.15, 0.15],
    [-0.3, 0.3],
    [-0.15, 0.15],
]  # 'YcLargeBins'
bins_lowHigh_list_dict["yc_bins_lowHigh_list"] = yc_bins_lowHigh_list
yc_bins_layers_list = make_bins_layers_list(yc_bins_lowHigh_list)
yc_bins_centers = yc_bins_layers_list[0]

fmm_bins_lowHigh_list = [
    [0.0, 0.0],
    [-0.2, 0.2],
    [-0.05, 0.05],
    [-0.05, 0.05],
    [-0.05, 0.05],
]  # percentage!!
bins_lowHigh_list_dict["fmm_bins_lowHigh_list"] = fmm_bins_lowHigh_list
fmm_bins_layers_list = make_bins_layers_list(fmm_bins_lowHigh_list)


v0_bins_lowHigh_list = [
    [0.0, 0.0],
    [-0.15, 0.15],
    [-0.05, 0.05],
    [-0.05, 0.05],
    [-0.05, 0.05],
]  # 'SmallerBins'
bins_lowHigh_list_dict["v0_bins_lowHigh_list"] = v0_bins_lowHigh_list
v0_bins_layers_list = make_bins_layers_list(v0_bins_lowHigh_list)


# human_bins = np.linspace(1., 2., 256)
human_bins = np.linspace(1.0, 1.9, 256)  # 'SmallerPersonBins'
# human_bins = np.linspace(1., 2.5, 256) #  'V2PersonCenBins'
# human_bins = np.linspace(0.7, 1.9, 256) #  'V3PersonCenBins'
human_bins_1 = np.linspace(-0.2, 0.2, 256)
human_bins_lowHigh_list = [
    [0.0, 0.0],
    [-0.3, 0.15],
    [-0.10, 0.10],
    [-0.10, 0.10],
    [-0.05, 0.05],
]  # 'SmallerBins'
bins_lowHigh_list_dict["human_bins_lowHigh_list"] = human_bins_lowHigh_list
human_bins_layers_list = make_bins_layers_list(human_bins_lowHigh_list)

car_bins = np.linspace(1.4, 1.70, 256)  # 'V2CarBins'
car_bins_lowHigh_list = [
    [0.0, 0.0],
    [-0.10, 0.10],
    [-0.05, 0.05],
    [-0.10, 0.10],
    [-0.05, 0.05],
]  # 'SmallerBins'
bins_lowHigh_list_dict["car_bins_lowHigh_list"] = car_bins_lowHigh_list
car_bins_layers_list = make_bins_layers_list(car_bins_lowHigh_list)

pitch_bins_low = np.linspace(-np.pi / 2 + 1e-5, -5 * np.pi / 180.0, 31)
pitch_bins_high = np.linspace(5 * np.pi / 180.0, np.pi / 6, 31)


def bin2midpointpitch(bins):
    pos = np.squeeze(bins.argmax(axis=-1))
    if pos < 31:
        return False, pitch_bins_low[pos]
    elif pos == 255:
        return False, np.pi / 6
    elif pos >= 224:
        return False, pitch_bins_high[pos - 224]
    else:
        return True, (pos - 32) / 192
