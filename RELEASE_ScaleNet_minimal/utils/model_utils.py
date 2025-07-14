import numpy as np
import torch
import torch.nn as nn
from dataset_coco_pickle_eccv import bins_lowHigh_list_dict
from dataset_coco_pickle_eccv import car_bins
from dataset_coco_pickle_eccv import car_bins_layers_list
from dataset_coco_pickle_eccv import fmm_bins_layers_list
from dataset_coco_pickle_eccv import human_bins
from dataset_coco_pickle_eccv import human_bins_layers_list
from dataset_coco_pickle_eccv import v0_bins_layers_list
from dataset_coco_pickle_eccv import yc_bins_centers
from dataset_coco_pickle_eccv import yc_bins_layers_list
from dataset_cvpr import horizon_bins_centers
from dataset_cvpr import pitch_bins_centers
from dataset_cvpr import roll_bins_centers
from dataset_cvpr import vfov_bins_centers
from maskrcnn_benchmark.structures.bounding_box import BoxList

pitch_bins_low = np.linspace(-np.pi / 2 + 1e-5, -5 * np.pi / 180.0, 31)
pitch_bins_high = np.linspace(5 * np.pi / 180.0, np.pi / 6, 31)
pitch_bins_v0_wide = np.concatenate(
    (np.linspace(-1.5, 0.0, 31), np.linspace(0.0, 1.0, 193), np.linspace(1.0, 1.5, 32)),
    0,
)

CATEGORIES = [
    "__background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def approx_model(input_dict):
    yc_est, vb, y_person, v0, vc, f_pixels_yannick = (
        input_dict["yc_est"],
        input_dict["vb"],
        input_dict["y_person"],
        input_dict["v0"],
        input_dict["vc"],
        input_dict["f_pixels_yannick"],
    )
    inv_f2 = 1.0 / (f_pixels_yannick * f_pixels_yannick)
    vt_camEst = (
        yc_est * vb + y_person * (v0 - vb) * (1.0 + inv_f2 * (vc - v0) * vc)
    ) / (yc_est + y_person * (v0 - vb) * inv_f2 * (vc - v0))
    return vt_camEst


def accu_model(input_dict, if_debug=False):
    negative_z = False
    yc_est, vb, y_person, v0, vc, f_pixels_yannick = (
        input_dict["yc_est"],
        input_dict["vb"],
        input_dict["y_person"],
        input_dict["v0"],
        input_dict["vc"],
        input_dict["f_pixels_yannick"],
    )
    # theta_yannick =  2 * torch.atan((vc - v0) / (2. * f_pixels_yannick))
    if "pitch_est" in input_dict:
        theta_yannick = input_dict["pitch_est"]
        if if_debug:
            print("Using pitch!", theta_yannick)
    else:
        theta_yannick = torch.atan((vc - v0) / f_pixels_yannick)
    if if_debug:
        print("---theta_yannick", theta_yannick)
    z = -(f_pixels_yannick * yc_est) / (
        f_pixels_yannick * torch.sin(theta_yannick)
        - (vc - vb) * torch.cos(theta_yannick)
        + 1e-10
    )
    print("---z", z)
    print("---z1", -(f_pixels_yannick * yc_est))
    print(
        "---z2",
        (
            f_pixels_yannick * torch.sin(theta_yannick)
            - (vc - vb) * torch.cos(theta_yannick)
            + 1e-10
        ),
    )

    if if_debug:
        print("---z", z)
    if z.detach().cpu().numpy() < 0:
        if if_debug:
            print(
                "----z----=====",
                z,
                theta_yannick / np.pi * 180.0,
                f_pixels_yannick,
                yc_est,
                theta_yannick,
                vc,
                vb,
            )
        negative_z = True
    vt_camEst = (
        (f_pixels_yannick * torch.cos(theta_yannick) + vc * torch.sin(theta_yannick))
        * y_person
        + (-f_pixels_yannick * torch.sin(theta_yannick) + vc * torch.cos(theta_yannick))
        * z
        + -f_pixels_yannick * yc_est
    ) / (y_person * torch.sin(theta_yannick) + z * torch.cos(theta_yannick) + 1e-10)
    return vt_camEst, z, negative_z


def accu_model_helanyi(input_dict, if_debug=False):
    negative_z = False
    yc_est, vb, y_person, v0, vc, f_pixels_yannick = (
        input_dict["yc_est"],
        input_dict["vb"],
        input_dict["y_person"],
        input_dict["v0"],
        input_dict["vc"],
        input_dict["f_pixels_yannick"],
    )
    theta_yannick = input_dict["pitch_est"]

    f = f_pixels_yannick
    uc = 0
    device = f.device
    cos = torch.cos(-theta_yannick)  # looking down: theta < 0
    sin = torch.sin(-theta_yannick)
    yc = yc_est
    y = y_person
    x = 0.0
    intrinsics = torch.tensor([[f, 0, uc], [0, f, vc], [0, 0, 1]]).to(device)
    Rt = torch.tensor(
        [[1, 0, 0, 0], [0, cos, sin, -yc * cos], [0, -sin, cos, yc * sin]],
    ).to(device)
    z = (-f * yc * cos + vc * yc * sin - (yc * sin) * vb) / (
        cos * vb - f * sin - vc * cos
    )
    xyz = torch.tensor([[x], [y], [z], [1.0]]).to(device)
    u_vt_1 = intrinsics @ Rt @ xyz
    vt_camEst = u_vt_1[1] / u_vt_1[2]
    return vt_camEst, z, negative_z


def accu_model_batch(input_dict, if_debug=False):
    yc_est, vb, y_person, v0, vc, f_pixels_yannick = (
        input_dict["yc_est"],
        input_dict["vb"],
        input_dict["y_person"],
        input_dict["v0"],
        input_dict["vc"],
        input_dict["f_pixels_yannick"],
    )
    if "pitch_est" in input_dict:
        theta_yannick = input_dict["pitch_est"]
        if if_debug:
            print("Using pitch!", theta_yannick)
    else:
        theta_yannick = torch.atan((vc - v0) / f_pixels_yannick)
    z = -(f_pixels_yannick * yc_est) / (
        f_pixels_yannick * torch.sin(theta_yannick)
        - (vc - vb) * torch.cos(theta_yannick)
        + 1e-10
    )
    vt_camEst = (
        (f_pixels_yannick * torch.cos(theta_yannick) + vc * torch.sin(theta_yannick))
        * y_person
        + (-f_pixels_yannick * torch.sin(theta_yannick) + vc * torch.cos(theta_yannick))
        * z
        + -f_pixels_yannick * yc_est
    ) / (y_person * torch.sin(theta_yannick) + z * torch.cos(theta_yannick) + 1e-10)
    negative_z = None
    return vt_camEst, z, negative_z


def get_pitch_est_bad(
    pitch_bins_low_device_batch,
    pitch_bins_mid_device_batch,
    pitch_bins_high_device_batch,
    output_pitch,
    batchsize,
    H_batch,
    vfov_estim,
    f_estim,
):
    half_fov_batch = (vfov_estim / 2.0).unsqueeze(-1)
    pitch_bins_low_device_batch_v0 = 0.5 - torch.tan(
        -pitch_bins_low_device_batch,
    ) * f_estim.unsqueeze(-1) / H_batch.unsqueeze(-1)
    pitch_bins_high_device_batch_v0 = 0.5 + torch.tan(
        pitch_bins_high_device_batch,
    ) * f_estim.unsqueeze(-1) / H_batch.unsqueeze(-1)
    pitch_bins_mid_device_batch_v0 = pitch_bins_mid_device_batch

    low_dims = pitch_bins_low_device_batch.shape[1]
    mid_dims = pitch_bins_mid_device_batch.shape[1]
    high_dims = pitch_bins_high_device_batch.shape[1]
    expectation_pitch_bins_low_device_batch_v0 = (
        torch.exp(nn.functional.log_softmax(output_pitch[:, :low_dims], dim=1))
        * pitch_bins_low_device_batch_v0
    ).sum(dim=1)
    expectation_pitch_bins_mid_device_batch_v0 = (
        torch.exp(
            nn.functional.log_softmax(
                output_pitch[:, low_dims : (low_dims + mid_dims)],
                dim=1,
            ),
        )
        * pitch_bins_mid_device_batch
    ).sum(dim=1)
    expectation_pitch_bins_high_device_batch_v0 = (
        torch.exp(nn.functional.log_softmax(output_pitch[:, -high_dims:], dim=1))
        * pitch_bins_high_device_batch_v0
    ).sum(dim=1)

    expectation_batch_v0 = (
        (torch.argmax(output_pitch, dim=1) < 31).float()
        * expectation_pitch_bins_low_device_batch_v0
        + (torch.argmax(output_pitch, dim=1) >= 224).float()
        * expectation_pitch_bins_high_device_batch_v0
        + (
            (torch.argmax(output_pitch, dim=1) >= 31).float()
            * (torch.argmax(output_pitch, dim=1) < 224).float()
        )
        * expectation_pitch_bins_mid_device_batch_v0
    )
    return expectation_batch_v0


def get_pitch_est(
    pitch_bins_low_device_batch,
    pitch_bins_mid_device_batch,
    pitch_bins_high_device_batch,
    output_pitch,
    batchsize,
    H_batch,
    vfov_estim,
    f_estim,
):
    half_fov_batch = (vfov_estim / 2.0).unsqueeze(-1)
    pitch_bins_low_device_batch_v0 = 0.5 - torch.tan(
        -pitch_bins_low_device_batch,
    ) * f_estim.unsqueeze(-1) / H_batch.unsqueeze(-1)
    pitch_bins_high_device_batch_v0 = 0.5 + torch.tan(
        pitch_bins_high_device_batch,
    ) * f_estim.unsqueeze(-1) / H_batch.unsqueeze(-1)
    pitch_bins_mid_device_batch_v0 = pitch_bins_mid_device_batch

    # expectation_pitch_bins_low_device_batch_v0 = pitch_bins_low_device_batch_v0 * output_pitch[:, :pitch_bins_low_device_batch.shape[0]]
    low_dims = pitch_bins_low_device_batch.shape[1]  # 31
    mid_dims = pitch_bins_mid_device_batch.shape[1]
    high_dims = pitch_bins_high_device_batch.shape[1]  # 32
    # print(output_pitch[:, :low_dims].shape, output_pitch[:, -high_dims:].shape, output_pitch[:, low_dims:(low_dims+mid_dims)].shape)
    expectation_pitch_bins_low_device_batch_v0 = (
        torch.exp(nn.functional.log_softmax(output_pitch[:, :low_dims], dim=1))
        * pitch_bins_low_device_batch_v0
    ).sum(dim=1)
    expectation_pitch_bins_mid_device_batch_v0 = (
        torch.exp(
            nn.functional.log_softmax(
                output_pitch[:, low_dims : (low_dims + mid_dims)],
                dim=1,
            ),
        )
        * pitch_bins_mid_device_batch
    ).sum(dim=1)
    expectation_pitch_bins_high_device_batch_v0 = (
        torch.exp(nn.functional.log_softmax(output_pitch[:, -high_dims:], dim=1))
        * pitch_bins_high_device_batch_v0
    ).sum(dim=1)

    expectation_batch_v0 = expectation_pitch_bins_mid_device_batch_v0
    return expectation_batch_v0


def get_pitch_est_v0(bins, output_pitch, H_batch, vfov_estim, f_estim):
    # half_fov_batch = (vfov_estim / 2.).unsqueeze(-1)
    pitch_bins_low_device_batch_v0 = 0.5 - torch.tan(
        -bins["pitch_bins_low_device_batch"],
    ) * f_estim.unsqueeze(-1) / H_batch.unsqueeze(-1)
    pitch_bins_high_device_batch_v0 = 0.5 + torch.tan(
        bins["pitch_bins_high_device_batch"],
    ) * f_estim.unsqueeze(-1) / H_batch.unsqueeze(-1)
    pitch_bins_mid_device_batch_v0 = bins["pitch_bins_mid_device_batch"].repeat(
        pitch_bins_low_device_batch_v0.shape[0],
        1,
    )
    pitch_bins_device_batch_v0 = torch.cat(
        (
            pitch_bins_low_device_batch_v0,
            pitch_bins_mid_device_batch_v0,
            pitch_bins_high_device_batch_v0,
        ),
        dim=1,
    )
    expectation_batch_v0 = (
        torch.exp(nn.functional.log_softmax(output_pitch, dim=1))
        * pitch_bins_device_batch_v0
    ).sum(dim=1)
    return expectation_batch_v0


def get_pitch_est_v0_mid(bins, output_pitch, H_batch, vfov_estim, f_estim):
    pitch_bins_mid_device_batch_v0 = bins["pitch_bins_mid_device_batch"].repeat(
        output_pitch.shape[0],
        1,
    )

    expectation_batch_v0 = (
        torch.exp(nn.functional.log_softmax(output_pitch[:, 31:224], dim=1))
        * pitch_bins_mid_device_batch_v0
    ).sum(dim=1)
    return expectation_batch_v0


def get_pitch_est_v0_wide(bins, output_pitch):
    # fixPitchv0
    expectation_batch_v0 = (
        torch.exp(nn.functional.log_softmax(output_pitch, dim=1))
        * bins["pitch_bins_v0_wide_device_batch"]
    ).sum(dim=1)
    return expectation_batch_v0


def get_horizon_est_yannick(bins, output_horizon):
    horizon_bins_device_batch = (
        bins["horizon_bins_centers_torch"]
        .unsqueeze(0)
        .repeat(output_horizon.shape[0], 1)
    )
    expectation_batch_horizon = (
        torch.exp(nn.functional.log_softmax(output_horizon, dim=1))
        * horizon_bins_device_batch
    ).sum(dim=1)
    return expectation_batch_horizon  # ([Yannick] 0 = top of the image, 1 = bottom of the image)


def get_pitch_radian_est_yannick(bins, output_pitch):
    pitch_bins_device_batch = (
        bins["pitch_bins_centers_torch"].unsqueeze(0).repeat(output_pitch.shape[0], 1)
    )
    expectation_batch_pitch = (
        torch.exp(nn.functional.log_softmax(output_pitch, dim=1))
        * pitch_bins_device_batch
    ).sum(dim=1)
    return expectation_batch_pitch  # ([Yannick] 0 = top of the image, 1 = bottom of the image)


def bin_mid_2midpointpitch(bins):
    pos = bins.argmax(dim=-1).float()
    return pos / 192.0


def get_bins(device):
    vfov_bins_centers_torch = torch.from_numpy(vfov_bins_centers).float().to(device)

    yc_bins_centers_torch = torch.from_numpy(yc_bins_centers).float().to(device)
    human_bins_torch = torch.from_numpy(human_bins).float().to(device)

    pitch_bins_low_device_batch = (
        torch.from_numpy(pitch_bins_low).float().to(device).unsqueeze(0)
    )
    pitch_bins_mid_device_batch = (
        torch.arange(0, 193).float().to(device) / 192
    ).unsqueeze(0)
    pitch_bins_high_device_batch = torch.cat(
        (
            torch.from_numpy(pitch_bins_high).float().to(device),
            torch.Tensor([np.pi / 6]).float().to(device),
        ),
    ).unsqueeze(0)
    pitch_bins_v0_wide_device_batch = (
        torch.from_numpy(pitch_bins_v0_wide).float().to(device).unsqueeze(0)
    )

    return {
        # 'roll_bins_centers_torch': roll_bins_centers_torch, 'distortion_bins_centers_torch': distortion_bins_centers_torch,
        "vfov_bins_centers_torch": vfov_bins_centers_torch,
        "yc_bins_centers_torch": yc_bins_centers_torch,
        "pitch_bins_low_device_batch": pitch_bins_low_device_batch,
        "pitch_bins_mid_device_batch": pitch_bins_mid_device_batch,
        "pitch_bins_high_device_batch": pitch_bins_high_device_batch,
        "pitch_bins_v0_wide_device_batch": pitch_bins_v0_wide_device_batch,
        "human_bins_torch": human_bins_torch,
    }


def get_bins_combine(device):
    bins_return = bins_lowHigh_list_dict.copy()

    vfov_bins_centers_torch = torch.from_numpy(vfov_bins_centers).float().to(device)
    roll_bins_centers_torch = torch.from_numpy(roll_bins_centers).float().to(device)
    pitch_bins_centers_torch = torch.from_numpy(pitch_bins_centers).float().to(device)
    horizon_bins_centers_torch = (
        torch.from_numpy(horizon_bins_centers).float().to(device)
    )

    yc_bins_centers_torch = torch.from_numpy(yc_bins_centers).float().to(device)
    human_bins_torch = torch.from_numpy(human_bins).float().to(device)
    car_bins_torch = torch.from_numpy(car_bins).float().to(device)

    yc_bins_layers_list_torch = [
        torch.from_numpy(yc_bins_layer).float().to(device)
        for yc_bins_layer in yc_bins_layers_list
    ]

    fmm_bins_layers_list_torch = [
        torch.from_numpy(fmm_bins_layer).float().to(device)
        for fmm_bins_layer in fmm_bins_layers_list
    ]

    v0_bins_layers_list_torch = [
        torch.from_numpy(v0_bins_layer).float().to(device)
        for v0_bins_layer in v0_bins_layers_list
    ]

    human_bins_layers_list_torch = [
        torch.from_numpy(human_bins_layer).float().to(device)
        for human_bins_layer in human_bins_layers_list
    ]

    car_bins_layers_list_torch = [
        torch.from_numpy(car_bins_layer).float().to(device)
        for car_bins_layer in car_bins_layers_list
    ]

    bins_return.update(
        {
            "vfov_bins_centers_torch": vfov_bins_centers_torch,
            "roll_bins_centers_torch": roll_bins_centers_torch,
            "pitch_bins_centers_torch": pitch_bins_centers_torch,
            "horizon_bins_centers_torch": horizon_bins_centers_torch,
            # 'yc_bins_centers_0_torch': yc_bins_centers_0_torch, 'yc_bins_centers_1_torch': yc_bins_centers_1_torch,
            "yc_bins_centers_torch": yc_bins_centers_torch,
            "human_bins_torch": human_bins_torch,  # 'human_bins_1_torch': human_bins_1_torch,
            "yc_bins_layers_list_torch": yc_bins_layers_list_torch,
            "fmm_bins_layers_list_torch": fmm_bins_layers_list_torch,
            "v0_bins_layers_list_torch": v0_bins_layers_list_torch,
            "human_bins_layers_list_torch": human_bins_layers_list_torch,
        },
    )
    bins_return.update(
        {
            "car_bins_torch": car_bins_torch,
            "car_bins_layers_list_torch": car_bins_layers_list_torch,
        },
    )

    return bins_return


def bboxArray_to_bboxList(
    bboxes_batch_array,
    bboxes_length_batch_array,
    W_batch_array,
    H_batch_array,
):
    bbox_list_list = []
    for bboxes_array, bboxes_length, W, H in zip(
        bboxes_batch_array,
        bboxes_length_batch_array,
        W_batch_array,
        H_batch_array,
    ):
        bbox_list = BoxList(bboxes_array[:bboxes_length, :], (W, H), "xywh")
        bbox_list = bbox_list.convert("xyxy")
        bbox_list_list.append(bbox_list)
    return bbox_list_list


def oneLargeBboxList(W_batch_array, H_batch_array):
    bbox_list_list = []
    for W, H in zip(W_batch_array, H_batch_array):
        # Following COCO annotations: (box coordinates are measured from the top left image corner and are 0-indexed) # http://cocodataset.org/#format-data
        bbox_list = BoxList(np.asarray([[0, 0, W, H]]), (W, H), "xywh")
        bbox_list = bbox_list.convert("xyxy")
        bbox_list_list.append(bbox_list)
    return bbox_list_list


def human_prior(tensor, mean=1.70, std=0.103):
    return (
        1.0
        / np.sqrt(2.0 * np.pi * (std**2))
        * torch.exp(-((tensor - mean) ** 2) / (2.0 * std**2))
    )
