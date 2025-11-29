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
    parser.add_argument(
        "--output-file",
        type=str,
        default="config/SUN360_train_crops_dataset_cvpr_myDistWider.json",
    )
    parser.add_argument("--ext", type=str, default="jpg", help="Filetype to filter for")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_file = args.output_file
    filetype = f"**/**/*.{args.ext}"
    pattern = os.path.join(input_dir, filetype)
    images = glob.glob(pattern)
    random.shuffle(images)
    with open(
        os.path.join(output_file),
        "w",
    ) as file:
        file.write(json.dumps(images))
