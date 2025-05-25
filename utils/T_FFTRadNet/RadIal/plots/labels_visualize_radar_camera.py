import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from utils.T_FFTRadNet.RadIal.ADCProcessing.DBReader.DBReader import SyncReader
from utils.T_FFTRadNet.RadIal.ADCProcessing.rpl import RadarSignalProcessing

# --- Constants and Camera Parameters ---
CAMERA_MATRIX = np.array([
    [1845.41929, 0.0, 855.802458],
    [0.0, 1788.69210, 607.342667],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([0.251771602, -13.2561698, 0.00433607564, -0.00694637533, 59.5513933])
RVEC = np.array([1.61803058, 0.03365624, -0.04003127])
TVEC = np.array([0.09138029, 1.38369885, 1.43674736])
IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080
INPUT_SHAPE = (540, 960)  # (height, width)
SCALE_FACTOR = (INPUT_SHAPE[1] / IMAGE_WIDTH, INPUT_SHAPE[0] / IMAGE_HEIGHT)


# --- Utility Functions ---
def get_scale_factor(image):
    """Return width and height scale factors for a given image."""
    height, width = image.shape[:2]
    return width / IMAGE_WIDTH, height / IMAGE_HEIGHT


def draw_bounding_box(image, bbox, label):
    """Draw a labeled bounding box on the image."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


def annotate_radar_point(ax, x, y):
    """Plot a radar point on the given axes."""
    ax.scatter(x, y, s=50, color='red', marker='o', alpha=0.7)


def convert_rd_coords(range_m, doppler_mps, shape):
    """Convert radar range and doppler to image coordinates."""
    height, width = shape
    y = np.clip(int((range_m / 103) * height), 0, height - 1)
    x = np.clip(int((doppler_mps / 0.1) + (width / 2)), 0, width - 1)
    return x, y


def convert_ra_coords(range_m, azimuth_deg, shape):
    """Convert radar range and azimuth to image coordinates."""
    height, width = shape
    y = np.clip(int((range_m / 103) * height), 0, height - 1)
    x = np.clip(int(((azimuth_deg + 90) / 180) * width), 0, width - 1)
    return x, y


# --- Visualization Function ---
def generate_figure(rd_path, ra_path, labels_df, image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale_w, scale_h = get_scale_factor(image)

    # Draw bounding boxes
    for _, row in labels_df.iterrows():
        bbox = (
            row['x1_pix'] * scale_w,
            row['y1_pix'] * scale_h,
            row['x2_pix'] * scale_w,
            row['y2_pix'] * scale_h,
        )
        label = f"R:{row['radar_R_m']:.1f} A:{row['radar_A_deg']:.1f} D:{row['radar_D_mps']:.1f}"
        image = draw_bounding_box(image, bbox, label)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Image with bounding boxes
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Camera Image with BBoxes")
    axs[0, 0].axis('off')

    # Load radar maps
    rd_map = np.load(rd_path)
    ra_map = np.load(ra_path)

    # Range-Doppler map
    axs[1, 0].imshow(rd_map, aspect='auto', origin='lower')
    axs[1, 0].invert_xaxis()
    axs[1, 0].set_title("Range-Doppler Map")

    # Axis labels for RD
    rd_height, rd_width = rd_map.shape
    doppler_ticks = np.linspace(-rd_width // 2 * 0.1, rd_width // 2 * 0.1, 5)
    x_ticks_rd = doppler_ticks / 0.1 + rd_width / 2
    axs[1, 0].set_xticks(x_ticks_rd)
    axs[1, 0].set_xticklabels([f"{t:.1f}" for t in doppler_ticks])
    axs[1, 0].set_xlabel("Doppler Velocity (m/s)")

    range_ticks = [0, 20, 40, 60, 80, 100]
    y_ticks_rd = [(r / 103) * rd_height for r in range_ticks]
    axs[1, 0].set_yticks(y_ticks_rd)
    axs[1, 0].set_yticklabels([f"{r}m" for r in range_ticks])
    axs[1, 0].set_ylabel("Range (m)")

    # Range-Azimuth map
    axs[1, 1].imshow(ra_map, aspect='auto', origin='lower')
    axs[1, 1].invert_xaxis()
    axs[1, 1].set_title("Range-Azimuth Map")

    ra_height, ra_width = ra_map.shape
    azimuth_ticks = [-90, -45, 0, 45, 90]
    x_ticks_ra = [((a + 90) / 180) * ra_width for a in azimuth_ticks]
    axs[1, 1].set_xticks(x_ticks_ra)
    axs[1, 1].set_xticklabels([f"{a}°" for a in azimuth_ticks])
    axs[1, 1].set_xlabel("Azimuth Angle (degrees)")

    y_ticks_ra = [(r / 103) * ra_height for r in range_ticks]
    axs[1, 1].set_yticks(y_ticks_ra)
    axs[1, 1].set_yticklabels([f"{r}m" for r in range_ticks])
    axs[1, 1].set_ylabel("Range (m)")

    # Annotate radar points
    for _, row in labels_df.iterrows():
        x_rd, y_rd = convert_rd_coords(row['radar_R_m'], row['radar_D_mps'], rd_map.shape)
        annotate_radar_point(axs[1, 0], x_rd, y_rd)

        x_ra, y_ra = convert_ra_coords(row['radar_R_m'], row['radar_A_deg'], ra_map.shape)
        annotate_radar_point(axs[1, 1], x_ra, y_ra)

    plt.tight_layout()
    return fig


# --- Main Pipeline ---
def main(labels_csv, image_dir, rd_dir, ra_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')
    dataset_names = labels_df['dataset'].unique()
    dataset_path = Path(image_dir).parent

    if not dataset_path.exists():
        print(f"Skipping datasets: directory {dataset_path} not found.")
        return

    for sample_id in labels_df['numSample'].unique():
        sample_labels = labels_df[labels_df['numSample'] == sample_id]

        image_path = Path(image_dir) / f"image_{sample_id:06d}.jpg"
        rd_path = Path(rd_dir) / f"rd_{sample_id:06d}.npy"
        ra_path = Path(ra_dir) / f"ra_{sample_id:06d}.npy"

        if not (image_path.exists() and rd_path.exists() and ra_path.exists()):
            print(f"Missing files for sample {sample_id}. Skipping.")
            continue

        fig = generate_figure(rd_path, ra_path, sample_labels, image_path)
        output_path = output_dir / f"viz_{sample_id:06d}.png"
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved visualization to {output_path}")


if __name__ == '__main__':
    import argparse

    path_repo = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils')
    if not path_repo.exists():
        path_repo = Path('/mnt/data/myprojects/PycharmProjects/thesis_repos/radar_sort/utils')

    path_config_default = path_repo / Path('T_FFTRadNet/RadIal/ADCProcessing/data_config.json')

    parser = argparse.ArgumentParser(description='Radar-Camera Visualization Script')
    parser.add_argument('-c', '--config', default=str(path_config_default),type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--output_dir', default=None, help='Directory to save output visualizations')

    args = parser.parse_args()

    config = json.load(open(args.config))

    record = config['target_value']
    root_folder = Path(config['Data_Dir'], 'RadIal_Data',record)
    labels_csv = Path(root_folder, 'labels.csv')
    image_dir = Path(root_folder, 'camera')
    rd_dir = Path(root_folder, 'radar_RD')
    ra_dir = Path(root_folder, 'radar_RA')

    # Prepare output folder structure
    output_dir = ra_dir.parent / 'output_visualizations'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)


    main(labels_csv, image_dir, rd_dir, ra_dir, output_dir)
