import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        default="quchenyuan/360x_dataset_LR",
    )
    parser.add_argument(
        "--allow-pattern",
        type=str,
        default="panoramic/*.mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/360x_dataset_LR",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()
    download_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        repo_type="dataset",
        allow_patterns=args.allow_pattern,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        for f in download_path:
            print(f)
