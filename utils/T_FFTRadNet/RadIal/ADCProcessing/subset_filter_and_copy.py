#!/usr/bin/env python3
import os
import sys
import shutil
import re
import argparse
import pandas as pd
import logging
from typing import List, Set

# Configure logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def read_labels(labels_path: str, datasets: List[str]) -> pd.DataFrame:
    """Reads labels.csv and filters rows based on dataset names."""
    try:
        labels_df = pd.read_csv(labels_path, sep=None, engine="python")
        filtered_labels = labels_df[labels_df["dataset"].isin(datasets)]
        if filtered_labels.empty:
            logging.warning("No matching dataset entries found.")
            sys.exit(0)
        return filtered_labels
    except Exception as e:
        logging.error(f"Failed to read labels file: {e}")
        sys.exit(1)

def extract_num_samples(labels_df: pd.DataFrame) -> Set[int]:
    """Extracts unique numSample values as integers."""
    try:
        return set(labels_df["numSample"].astype(int))
    except Exception as e:
        logging.error(f"Error processing 'numSample' column: {e}")
        sys.exit(1)

def create_directory_structure(base_dir: str, subdirs: List[str]) -> None:
    """Creates the output directory structure."""
    os.makedirs(base_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    logging.info(f"Created output directory structure in: {base_dir}")

def copy_relevant_files(root_dir: str, output_root: str, num_samples: Set[int], subdirs: List[str]) -> None:
    """Copies files matching the filtered numSample values into the output directory."""
    for sub in subdirs:
        src_subdir = os.path.join(root_dir, sub)
        dst_subdir = os.path.join(output_root, sub)

        if not os.path.exists(src_subdir):
            logging.warning(f"Skipping non-existent directory: {src_subdir}")
            continue

        for filename in os.listdir(src_subdir):
            match = re.search(r"(\d+)", filename)
            if match:
                file_num = int(match.group(1))
                if file_num in num_samples:
                    shutil.copy2(os.path.join(src_subdir, filename), os.path.join(dst_subdir, filename))
                    logging.info(f"Copied {filename} to {dst_subdir}")

def save_filtered_labels(labels_df: pd.DataFrame, output_path: str) -> None:
    """Saves the filtered labels.csv file."""
    labels_df.to_csv(output_path, index=False)
    logging.info(f"Filtered labels.csv saved at {output_path}")

def main():
    path_root =  "/mnt/data/datasets/radial/gd/RADIal"
    name_dataset = ["RECORD@2020-11-22_12.37.16"]
    path_labels = "/mnt/data/datasets/radial/gd/raw_data/labels_CVPR.csv"
    path_output = "/mnt/data/datasets/radial/gd/subset"


    """Main function to process the datasets."""
    parser = argparse.ArgumentParser(description="Filter and copy RadIal data based on dataset names and numSample values.")
    parser.add_argument("--root", type=str, default=path_root, help="Path to the original RadIal_Data directory.")
    parser.add_argument("--datasets", type=str, default=name_dataset, nargs="+", help="List of dataset names to filter.")
    parser.add_argument("--labels", type=str, default=path_labels, help="Path to labels.csv file.")
    parser.add_argument("--output", type=str, default=path_output, help="Path to the output directory.")

    args = parser.parse_args()

    subdirs = ["ADC_Data", "camera", "laser_PCL", "radar_FFT", "radar_Freespace", "radar_PCL"]

    # Read and filter labels
    labels_df = read_labels(args.labels, args.datasets)
    num_samples = extract_num_samples(labels_df)

    # Create output directory structure
    create_directory_structure(args.output, subdirs)

    # Copy relevant files
    copy_relevant_files(args.root, args.output, num_samples, subdirs)

    # Save filtered labels.csv
    save_filtered_labels(labels_df, os.path.join(args.output, "labels.csv"))

if __name__ == "__main__":
    main()

# python subset_filter_and_copy.py --root /path/to/RadIal_Data --datasets RECORD@2020-11-21_13.44.44 another_dataset --labels /path/to/RadIal_Data/labels.csv --output /path/to/output_folder
# python subset_filter_and_copy.py --root /mnt/data/datasets/radial/gd/RADIal --datasets RECORD@2020-11-22_12.20.19 --labels /mnt/data/datasets/radial/gd/RADIal/labels.csv --output /mnt/data/datasets/radial/gd/subset