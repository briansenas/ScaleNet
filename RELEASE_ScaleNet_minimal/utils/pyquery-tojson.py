import argparse
import json
import os
import random

seed = 140421
random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/Flicker360/Flicker360-reviewed.json",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="config/Flicker360_reviewed_train_crops_dataset_cvpr_myDistWider.json",
    )
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    with open(
        os.path.join(input_file),
        "r",
    ) as file:
        data = json.loads(file.read())

    with open(output_file, "w") as file:
        file.write(json.dumps(list(data.keys())))
