import json
import logging
import os
import random
import resource
import sys
from glob import glob

import numpy as np
import torch
from panorama_cropping_dataset_generation.debugging import getHorizonLine
from PIL import Image
from termcolor import colored
from utils.utils_coco import getBins

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

pitch_bins_low = np.linspace(-np.pi / 2 + 1e-5, -5 * np.pi / 180.0, 31)
pitch_bins_high = np.linspace(5 * np.pi / 180.0, np.pi / 6, 31)

# DS_ROOT = "/home/holdgeof/data/dimension_match_image_crops/"
# DS_ROOT = "/home/holdgeof/data/sun360_sphericaldistortion"
# DS_ROOT = "/newfoundland/data_extra/SUN360/crops_dataset_old"
# DS_ROOT = "data/SUN360/sun360_sphericaldistortion_oriDist_full"
# DS_ROOT = 'data/SUN360/crops_dataset_kalyan'
# DS_ROOT = 'data/SUN360/crops_dataset_cvpr_myDist'


# DS_ROOT = 'data/SUN360/crops_dataset_cvpr_myDistNarrowerLarge1105' # ECCV submission: SUNV1
DS_ROOT = "data/SUN360_mini_crops_dataset_cvpr_myDistNarrowerLarge1105"
# DS_ROOT = 'data/SUN360/crops_dataset_cvpr_myDistWider20200403' # SUNV2

if "Narrower" in DS_ROOT:
    # crops_dataset_cvpr_myDistNarrower
    print(colored("[data-SUN360] Narrower in DS_ROOT:" + DS_ROOT, "white", "on_blue"))
    pitch_bins = np.linspace(-0.6, 0.6, 255)
    pitch_bins_centers = pitch_bins.copy()
    pitch_bins_centers[:-1] += np.diff(pitch_bins_centers) / 2
    pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

    horizon_bins = np.linspace(-0.5, 0.9, 255)
    horizon_bins_centers = horizon_bins.copy()
    horizon_bins_centers[:-1] += np.diff(horizon_bins_centers) / 2
    horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

    roll_bins = getBins(-np.pi / 6, np.pi / 6, 0.5, 0.04, 1.1, np.pi)
    roll_bins_centers = roll_bins.copy()
    roll_bins_centers[:-1] += np.diff(roll_bins_centers) / 2
    roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

    vfov_bins = np.linspace(0.2389, 1.6, 255)
    vfov_bins_centers = vfov_bins.copy()
    vfov_bins_centers[:-1] += np.diff(vfov_bins_centers) / 2
    vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])

elif "Wider20200403" in DS_ROOT:
    print(
        colored(
            "[data-SUN360] Wider20200403 in DS_ROOT:" + DS_ROOT,
            "white",
            "on_blue",
        ),
    )
    # crops_dataset_cvpr_myDistWider20200403:
    pitch_bins = np.linspace(-0.6, 0.6, 255)
    pitch_bins_centers = pitch_bins.copy()
    pitch_bins_centers[:-1] += np.diff(pitch_bins_centers) / 2
    pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

    horizon_bins = np.linspace(-1.0, 0.95, 255)
    horizon_bins_centers = horizon_bins.copy()
    horizon_bins_centers[:-1] += np.diff(horizon_bins_centers) / 2
    horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

    roll_bins = getBins(-np.pi / 6, np.pi / 6, 0.5, 0.04, 1.1, np.pi)
    roll_bins_centers = roll_bins.copy()
    roll_bins_centers[:-1] += np.diff(roll_bins_centers) / 2
    roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

    vfov_bins = np.linspace(0.2389, 1.6, 255)
    vfov_bins_centers = vfov_bins.copy()
    vfov_bins_centers[:-1] += np.diff(vfov_bins_centers) / 2
    vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])

else:
    print(colored("[data-SUN360] other in DS_ROOT:" + DS_ROOT, "white", "on_blue"))
    # crops_dataset_kalyan || crops_dataset_cvpr_myDist
    pitch_bins = np.linspace(-0.6, 0.6, 255)
    pitch_bins_centers = pitch_bins.copy()
    pitch_bins_centers[:-1] += np.diff(pitch_bins_centers) / 2
    pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

    horizon_bins = np.linspace(-0.5, 1.5, 255)
    horizon_bins_centers = horizon_bins.copy()
    horizon_bins_centers[:-1] += np.diff(horizon_bins_centers) / 2
    horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

    roll_bins = getBins(-np.pi / 6, np.pi / 6, 0.5, 0.04, 1.1, np.pi)
    roll_bins_centers = roll_bins.copy()
    roll_bins_centers[:-1] += np.diff(roll_bins_centers) / 2
    roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

    vfov_bins = np.linspace(0.2389, 1.6, 255)
    vfov_bins_centers = vfov_bins.copy()
    vfov_bins_centers[:-1] += np.diff(vfov_bins_centers) / 2
    vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])


def getHorizonLineFromAngles(pitch, roll, FoV, im_h, im_w):
    midpoint = getMidpointFromAngle(pitch, FoV)
    dh = getDeltaHeightFromRoll(roll, im_h, im_w)
    return midpoint + dh, midpoint - dh


def getMidpointFromAngle(pitch, FoV):
    return 0.5 + 0.5 * np.tan(pitch) / np.tan(FoV / 2)


def getDeltaHeightFromRoll(roll, im_h, im_w):
    "The height distance of horizon from the midpoint at image left/right border intersection." ""
    return im_w / im_h * np.tan(roll) / 2


def getOffset(pitch, roll, vFoV, im_h, im_w):
    hl, hr = getHorizonLineFromAngles(pitch, roll, vFoV, im_h, im_w)
    midpoint = (hl + hr) / 2.0
    # slope = np.arctan(hr - hl)
    offset = (midpoint - 0.5) / np.sqrt(1 + (hr - hl) ** 2)
    return offset


def midpointpitch2bin(midpoint, pitch):
    if np.isnan(midpoint):
        if pitch < 0:
            return np.digitize(pitch, pitch_bins_low)
        else:
            return np.digitize(pitch, pitch_bins_high) + 224
    assert 0 <= midpoint <= 1
    return int(midpoint * 192) + 32


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


def bins2horizon(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return horizon_bins_centers[idxes]


def bins2pitch(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return pitch_bins_centers[idxes]


def bins2roll(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return roll_bins_centers[idxes]


def bins2vfov(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return vfov_bins_centers[idxes]


class SUN360Horizon:
    def __init__(self, transforms=None, train=True, logger=None):

        self.transforms = transforms
        if logger is None:
            self.logger = logging.getLogger("SUN360Horizon")
        else:
            self.logger = logger
        import time

        ts = time.time()
        json_name = DS_ROOT.split("/")[-1]
        try:
            with open("%s.json" % json_name) as fhdl:
                self.data = json.load(fhdl)
        except FileNotFoundError:
            self.data = glob(os.path.join(DS_ROOT, "**/*.jpg"), recursive=True)
            with open("%s.json" % json_name, "w") as fhdl:
                json.dump(self.data, fhdl)
        self.logger.info(
            colored(
                "[SUN360 dataset] Loaded %d images from %s in %.2f seconds."
                % (len(self.data), DS_ROOT, time.time() - ts),
                "white",
                "on_blue",
            ),
        )

        random.seed(314159265)
        random.shuffle(self.data)
        if train:
            self.data = self.data[:-2000]
        else:
            self.data = self.data[-2000:]

        self.logger.info(
            "===== %d for the %s set..."
            % (len(self.data), "TRAIN" if train else "VAL"),
        )

    def __getitem__(self, k):
        with open(self.data[k][:-4] + ".json") as fhdl:
            data = json.load(fhdl)
        im_path = self.data[k]
        im_ori_RGB = Image.open(im_path).convert("RGB")  # im_ori_RGB.size: [W, H]
        assert "Narrower" in DS_ROOT or "DistWider20200403" in DS_ROOT
        data = data[0]
        pitch = data["pitch"]  # in radians
        roll = data["roll"]
        vfov = data["vfov"]
        focal_length_35mm_eq = data["focal_length_35mm_eq"]
        horizon = getHorizonLine(vfov, pitch)
        sensor_size = data["sensor_size"]
        idx1 = np.digitize(horizon, horizon_bins)
        idx2 = np.digitize(pitch, pitch_bins)
        idx3 = np.digitize(roll, roll_bins)
        idx4 = np.digitize(vfov, vfov_bins)

        y1 = np.zeros((256,), dtype=np.float32)
        y2 = np.zeros((256,), dtype=np.float32)
        y3 = np.zeros((256,), dtype=np.float32)
        y4 = np.zeros((256,), dtype=np.float32)

        y1[idx1] = y2[idx2] = y3[idx3] = y4[idx4] = 1.0

        im = self.transforms(im_ori_RGB)
        y1, y2, y3, y4 = map(torch.from_numpy, (y1, y2, y3, y4))

        W_ori, H_ori = im_ori_RGB.size

        return (
            im_path,
            im,
            y1,
            y2,
            y3,
            y4,
            data,
            pitch,
            roll,
            vfov,
            horizon,
            focal_length_35mm_eq,
            sensor_size,
            W_ori,
            H_ori,
            idx1,
            idx2,
            idx3,
            idx4,
        )

    def __len__(self):
        return len(self.data)


def my_collate_SUN360(batch):
    # Refer to https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    (
        im_path_list,
        im_list,
        y1_list,
        y2_list,
        y3_list,
        y4_list,
        data_list,
        pitch_list,
        roll_list,
        vfov_list,
        horizon_list,
        focal_length_35mm_eq_list,
        sensor_size_list,
        W_list,
        H_list,
        idx1_list,
        idx2_list,
        idx3_list,
        idx4_list,
    ) = zip(*batch)

    y1 = torch.stack(y1_list)
    y2 = torch.stack(y2_list)
    y3 = torch.stack(y3_list)
    y4 = torch.stack(y4_list)

    idx1 = torch.tensor(idx1_list)
    idx2 = torch.tensor(idx2_list)
    idx3 = torch.tensor(idx3_list)
    idx4 = torch.tensor(idx4_list)

    return (
        im_path_list,
        im_list,
        y1,
        y2,
        y3,
        y4,
        data_list,
        pitch_list,
        roll_list,
        vfov_list,
        horizon_list,
        focal_length_35mm_eq_list,
        sensor_size_list,
        W_list,
        H_list,
        idx1,
        idx2,
        idx3,
        idx4,
    )


if __name__ == "__main__":

    this_bin = midpointpitch2bin(1.1, 0.0)
    a = np.zeros((256,))
    a[this_bin] = 1
    print("bin:", this_bin, "recovered:", bin2midpointpitch(a))

    sys.exit()
    train = SUN360Horizon(train=True)
    print(len(train))
    for a in range(len(train)):
        _ = train[a]
