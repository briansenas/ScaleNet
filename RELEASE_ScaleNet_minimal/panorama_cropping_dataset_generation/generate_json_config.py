import argparse
import glob
import json
import os
import random

seed = 140421
random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/SUN360/train_crops_dataset_cvpr_myDistWider",
    )
    parser.add_argument("--output-dir", type=str, default="config")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    images = glob.glob(os.path.join(input_dir, "*/*.jpg"))
    random.shuffle(images)
    with open(
        os.path.join(output_dir, os.path.basename(input_dir) + ".json"),
        "w",
    ) as file:
        file.write(json.dumps(images))
