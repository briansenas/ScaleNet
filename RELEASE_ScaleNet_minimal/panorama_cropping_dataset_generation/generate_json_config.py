import argparse
import glob
import json
import os
import random
from sklearn.model_selection import train_test_split

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
        "--split",
        action="store_true",
        help="Whether or not to do train test split",
    )
    parser.add_argument(
        "--val-split",
        action="store_true",
        help="Whether or not to do train test split",
    )
    parser.add_argument("--test-size", default="0.1", help="Portion of the test size")
    parser.add_argument(
        "--val-size", default="0.1", help="Portion of the validation size"
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
    filetype = f"**/*.{args.ext}"
    pattern = os.path.join(input_dir, filetype)
    print("Using the glob pattern: ", pattern)
    images = glob.glob(pattern)
    print("Number of files found:", len(images))
    images = list(filter(lambda x: "debug" not in x, images))
    print("Filter debug files: ", len(images))
    if not args.split:
        random.shuffle(images)
        with open(
            os.path.join(output_file),
            "w",
        ) as file:
            file.write(json.dumps(images))
    else:
        train, test = train_test_split(
            images, test_size=float(args.test_size), shuffle=True, random_state=140421
        )
        files = [train, test]
        names = ["train", "test"]
        if args.val_split:
            train, val = train_test_split(
                train, test_size=float(args.val_size), shuffle=True, random_state=140421
            )
            files.append(val)
            names.append("val")
        for data, name in zip([train, test], ["train", "test"]):
            with open(
                os.path.join(output_file + "_" + name + ".json"),
                "w",
            ) as file:
                file.write(json.dumps(data))
