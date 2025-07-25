import argparse
import json
import os
import random
from glob import glob
from multiprocessing import Pool

import numpy as np
import panorama_cropping_dataset_generation.image_extraction as image_extraction
from panorama_cropping_dataset_generation.debugging import showHorizonLine
from panorama_cropping_dataset_generation.helpers import DispDebug
from PIL import Image
from scipy.stats import cauchy
from scipy.stats import lognorm
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm


DEBUG = False
DISPLAY = False


aspect_ratios, ar_probabilities = zip(
    *(
        (1 / 1, 0.09),  # Old cameras, Polaroids and Instagram
        (5 / 4, 0.01),  # Large and medium format photography, 8x10 picture frames
        (4 / 3, 0.66),  # Most point-and-shoots, Four-Thirds cameras,
        (3 / 2, 0.20),  # 35mm cameras, D-SLR
        (16 / 9, 0.04),  # Cameras mimicking the 16:9 widescreen format
    ),
)
assert sum(ar_probabilities) == 1


# horizon_mu, horizon_sigma, horizon_lower, horizon_upper = 0.523, 0.152, -0.1, 5
# roll_mu, roll_sigma, roll_lower, roll_upper = 0.0, 0.020, -8, 8
# focal_mu, focal_sigma, focal_lower, focal_upper = 14, 17, 1, 100

# Original
# horizon_mu, horizon_sigma, horizon_lower, horizon_upper = 0.523, 0.3, -1, 5
# roll_mu, roll_sigma, roll_sigma_low, roll_lower, roll_upper = 0.0, 0.1, 0.001, -np.pi/4, np.pi/4
# focal_mu, focal_sigma, focal_lower, focal_upper = 14, 17, 1, 100

# 'myDist'
# horizon_mu, horizon_sigma, horizon_lower, horizon_upper = 0.523, 0.3, -1, 2
# roll_mu, roll_sigma, roll_sigma_low, roll_lower, roll_upper = 0.0, 0.1, 0.001, -np.pi/6, np.pi/6
# focal_mu, focal_sigma, focal_lower, focal_upper = 14, 17, 1, 100

# 'myDistNarrower'
horizon_mu, horizon_sigma, horizon_lower, horizon_upper = 0.523, 0.3, -0.5, 0.9
roll_mu, roll_sigma, roll_sigma_low, roll_lower, roll_upper = (
    0.0,
    0.1,
    0.001,
    -np.pi / 6,
    np.pi / 6,
)
focal_mu, focal_sigma, focal_lower, focal_upper = 14, 17, 12, 100

# 'myDistWider20200403' # SUNV2
horizon_mu, horizon_sigma, horizon_lower, horizon_upper = 0.523, 0.3, -1.0, 0.95
roll_mu, roll_sigma, roll_sigma_low, roll_lower, roll_upper = (
    0.0,
    0.1,
    0.001,
    -np.pi / 6,
    np.pi / 6,
)
focal_mu, focal_sigma, focal_lower, focal_upper = 14, 17, 12, 100

portrait_probability = [0.80, 0.20]


def getHalfFoV(sensor_size, f):
    """
    :sensor_size: sensor size, could contain 2 numbers or more.
    :f: Focal length, number of elements must match `sensor_size`
    """
    return np.arctan2(sensor_size, (2 * f))


def makeAndSaveImg(img_id, img, rndid, if_debug=False):
    sensor_size = 24  # 35mm format is 36x24
    yaw = np.random.uniform(-np.pi, np.pi)
    ar = np.asscalar(np.random.choice(aspect_ratios, p=ar_probabilities))

    pitch = np.inf
    focal_length = np.inf
    while not focal_lower < focal_length < focal_upper:
        focal_length = np.clip(
            lognorm.rvs(s=0.8, loc=focal_mu, scale=focal_sigma),
            focal_lower,
            focal_upper,
        )
    horizon = np.random.normal(horizon_mu, horizon_sigma)
    while not horizon_lower < horizon < horizon_upper:
        horizon = np.random.normal(horizon_mu, horizon_sigma)

    low_roll = np.random.choice((True, False), p=(0.33, 0.67))
    roll = np.inf
    while not roll_lower < roll < roll_upper:
        if low_roll:
            roll = cauchy.rvs(loc=roll_mu, scale=roll_sigma_low, size=1)[0]
        else:
            roll = cauchy.rvs(loc=roll_mu, scale=roll_sigma, size=1)[0]

    vfov = 2 * getHalfFoV(sensor_size, focal_length)

    fl_px = focal_length / sensor_size
    pitch = -np.arctan((horizon - 0.5) / fl_px)

    is_portrait = bool(
        np.asscalar(np.random.choice(np.arange(2), p=portrait_probability)),
    )

    if is_portrait:
        ar = 1 / ar
        sensor_size = 36  # 35mm format is 36x24
        vfov = 2 * getHalfFoV(sensor_size, focal_length)

    # resY = 256
    resY = 600
    resX = int(resY / ar)

    if resX < 256:
        resX = 256
        resY = int(resX * ar)

    if if_debug:
        print(
            f"{img_id}\tangles ({np.degrees(pitch)}, {np.degrees(yaw)}, {np.degrees(roll)})\tres {resX}x{resY}\tvFOV {np.degrees(vfov)} ({focal_length}mm)\taspect {ar}",
        )

    im = image_extraction.extractImage(
        img,
        [pitch, yaw, roll],
        resY,
        vfov=vfov * 180 / np.pi,
        ratio=ar,
    )

    im_filename = f"{os.path.basename(img_id)}-{rndid}.jpg"
    subdir = im_filename.replace("pano_a", "")[:2]
    # imsave(im_filename), im)
    im_pil = Image.fromarray(im)
    im_pil.save(
        os.path.join(output_dir, subdir, im_filename),
        quality=95,
        optimize=False,
        progressive=False,
    )

    if DEBUG or np.random.random() < 0.001:
        im, _ = showHorizonLine(
            im,
            vfov,
            pitch,
            roll,
            focal_length=focal_length,
            debug=True,
        )
        save_file_path = os.path.join(output_dir, "debug", im_filename)
        imsave(save_file_path, im)
        print("Debug image saved to " + save_file_path)

    if DISPLAY:
        import matplotlib

        matplotlib.use("qt5agg")
        from matplotlib import pyplot as plt

        plt.clf()
        plt.imshow(im)
        plt.show(block=False)
        plt.pause(0.001)

    data_tosave = {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "vfov": vfov,
        "focal_length_35mm_eq": focal_length,
        "f_px": fl_px,
        "height": resX,
        "width": resY,
        "sensor_size": sensor_size,
        "horizon": horizon,
    }

    return (
        data_tosave,
        img_id,
        im_filename,
        yaw,
        pitch,
        roll,
        vfov,
        focal_length,
        resX,
        resY,
        sensor_size,
        horizon,
    )


def randomize():
    np.random.seed()


def process(im_path):
    im_filename = f"{os.path.basename(im_path)}-{0}.jpg"
    subdir = im_filename.replace("pano_a", "")[:2]
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    out_path = os.path.join(output_dir, subdir, im_filename)

    metadata_path = os.path.join(
        output_dir,
        subdir,
        f"{os.path.basename(im_path)}-meta.json",
    )
    if os.path.isfile(metadata_path):
        return

    im = imread(im_path)
    if len(im.shape) == 2:
        im = np.stack((im,) * 3, axis=-1)
    else:
        im = im[:, :, :3]
    data = []
    for random_id in range(7):
        datum = makeAndSaveImg(im_path, im, random_id)
        json_path = os.path.splitext(out_path)[0][:-2] + "-%d.json" % random_id
        with open(json_path, "w") as fhdl_datum:
            json.dump(datum, fhdl_datum)
        data.append(datum)

    with open(metadata_path, "w") as fhdl:
        json.dump(data, fhdl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/SUN360/train/RGB",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/SUN360/train_crops_dataset_cvpr_myDistWider",
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)
    with DispDebug("Listing SUN360..."):
        images = glob(input_dir + "/**/*.jpg", recursive=True)

    random.seed(31415926)
    random.shuffle(images)
    with Pool(processes=4, initializer=randomize) as pool:
        for _ in list(tqdm(pool.imap_unordered(process, images), total=len(images))):
            pass
