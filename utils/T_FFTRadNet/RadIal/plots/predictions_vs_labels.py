import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from utils.T_FFTRadNet.RadIal.utils.util import worldToImage

from dl_output_viz import visualize_detections_on_bev

class RadarVisualizationTool:
    def __init__(self, camera_params=None):
        # Default camera parameters
        self.camera_matrix = np.array([
            [1845.41929, 0.0, 855.802458],
            [0.0, 1788.69210, 607.342667],
            [0.0, 0.0, 1.0]
        ]) if camera_params is None else camera_params['matrix']

        self.image_width, self.image_height = 1920, 1080

    def load_data(self, labels_csv, predictions_csv):
        """Load labels and predictions from CSV files"""
        labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')
        predictions_df = pd.read_csv(predictions_csv)
        return labels_df, predictions_df

    def get_scale_factor(self, image):
        """Calculate scale factors for image"""
        height, width = image.shape[:2]
        return width / self.image_width, height / self.image_height

    def draw_bounding_box(self, image, bbox, label, color=(0, 255, 0), thickness=2):
        """Draw labeled bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return image

    def convert_ra_coords(self, range_m, azimuth_deg, shape):
        """Convert range-azimuth to image coordinates"""
        height, width = shape
        y = np.clip(int((range_m / 103) * height), 0, height - 1)
        x = np.clip(int(((azimuth_deg + 90) / 180) * width), 0, width - 1)
        return x, y

    def visualize_sample(self, sample_id, labels_df, predictions_df,
                         image_path, rd_path, ra_path):
        """Create visualization for single sample"""
        # Load data
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rd_map = np.load(rd_path)
        ra_map = np.load(ra_path)

        # Filter data for this sample
        sample_labels = labels_df[labels_df['numSample'] == sample_id]
        sample_predictions = predictions_df[predictions_df['sample_id'] == sample_id]

        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # Camera image with annotations
        image_viz = self._annotate_camera_image(
            image.copy(), sample_labels, sample_predictions
        )

        axs[0, 0].imshow(image_viz)
        axs[0, 0].set_title(f"Sample {sample_id}: Green=GT, Red=Net Output")
        axs[0, 0].axis('off')

        # BEV - Bird Eye View
        used_cols = ['confidence', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'range_m', 'azimuth_deg']
        bev_img = visualize_detections_on_bev(ra_map, sample_predictions[used_cols].to_numpy())
        axs[0, 1].imshow(bev_img)
        axs[0, 1].set_title(f"Pill View: Net Output")
        axs[0, 1].axis('off')

        # Range-Doppler map
        axs[1, 0].imshow(rd_map, aspect='auto', origin='lower')
        axs[1, 0].invert_xaxis()
        axs[1, 0].set_title("Range-Doppler Map")
        self._setup_rd_axes(axs[1, 0], rd_map.shape)

        # Range-Azimuth map with points
        axs[1, 1].imshow(ra_map, aspect='auto', origin='lower')
        axs[1, 1].invert_xaxis()
        axs[1, 1].set_title("Range-Azimuth: Green=GT, Red=Net Output")
        self._setup_ra_axes(axs[1, 1], ra_map.shape)
        self._annotate_ra_map(axs[1, 1], sample_labels, sample_predictions, ra_map.shape)

        plt.tight_layout()
        return fig

    def _annotate_camera_image(self, image, labels_df, predictions_df):
        """Add bounding boxes to camera image"""

        # Ground truth (green) - with scaling
        for _, row in labels_df.iterrows():
            scale_w, scale_h = self.get_scale_factor(image)
            bbox = (
                row['x1_pix'] * scale_w,
                row['y1_pix'] * scale_h,
                row['x2_pix'] * scale_w,
                row['y2_pix'] * scale_h,
            )
            label = f"GT R:{row['radar_R_m']:.1f} A:{row['radar_A_deg']:.1f}"
            image = self.draw_bounding_box(image, bbox, label, color=(0, 255, 0))

        # Predictions (red) - convert from world to image coordinates
        for _, row in predictions_df.iterrows():
            # Convert range/azimuth to cartesian
            range_m = row['range_m']
            azimuth_deg = row['azimuth_deg']
            x = np.sin(np.deg2rad(azimuth_deg)) * range_m
            y = np.cos(np.deg2rad(azimuth_deg)) * range_m

            # Project to image coordinates using your worldToImage function
            u1, v1 = worldToImage(-x - 0.9, y, 0)  # Front-left corner
            u2, v2 = worldToImage(-x + 0.9, y, 1.6)  # Back-right corner

            # Scale coordinates (divide by 2 as in your working example)
            u1, v1 = int(u1 / 2), int(v1 / 2)
            u2, v2 = int(u2 / 2), int(v2 / 2)

            bbox = (u1, v1, u2, v2)
            label = f"P R:{range_m:.1f} A:{azimuth_deg:.1f} C:{row['confidence']:.2f}"
            image = self.draw_bounding_box(image, bbox, label, color=(255, 0, 0), thickness=3)

        return image

    def _annotate_ra_map(self, ax, labels_df, predictions_df, ra_shape):
        """Add points to RA map"""
        # Ground truth (green circles)
        for _, row in labels_df.iterrows():
            x, y = self.convert_ra_coords(row['radar_R_m'], row['radar_A_deg'], ra_shape)
            ax.scatter(x, y, s=100, color='green', marker='o', alpha=0.8)

        # Predictions (red X)
        for _, row in predictions_df.iterrows():
            x, y = self.convert_ra_coords(row['range_m'], row['azimuth_deg'], ra_shape)
            ax.scatter(x, y, s=100, color='red', marker='x', alpha=0.8)

    def _setup_rd_axes(self, ax, shape):
        """Setup Range-Doppler axes"""
        height, width = shape
        doppler_ticks = np.linspace(-width // 2 * 0.1, width // 2 * 0.1, 5)
        x_ticks = doppler_ticks / 0.1 + width / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in doppler_ticks])
        ax.set_xlabel("Doppler Velocity (m/s)")

        range_ticks = [0, 20, 40, 60, 80, 100]
        y_ticks = [(r / 103) * height for r in range_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{r}m" for r in range_ticks])
        ax.set_ylabel("Range (m)")

    def _setup_ra_axes(self, ax, shape):
        """Setup Range-Azimuth axes"""
        height, width = shape
        azimuth_ticks = [-90, -45, 0, 45, 90]
        x_ticks = [((a + 90) / 180) * width for a in azimuth_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{a}°" for a in azimuth_ticks])
        ax.set_xlabel("Azimuth Angle (degrees)")

        range_ticks = [0, 20, 40, 60, 80, 100]
        y_ticks = [(r / 103) * height for r in range_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{r}m" for r in range_ticks])
        ax.set_ylabel("Range (m)")


def main(labels_csv, predictions_csv, image_dir, rd_dir, ra_dir, output_dir):
    """Main visualization pipeline"""
    viz_tool = RadarVisualizationTool()
    labels_df, predictions_df = viz_tool.load_data(labels_csv, predictions_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all sample IDs from labels
    sample_ids = labels_df['numSample'].unique()

    for sample_id in sample_ids:
        # File paths
        image_path = Path(image_dir) / f"image_{sample_id:06d}.jpg"
        rd_path = Path(rd_dir) / f"rd_{sample_id:06d}.npy"
        ra_path = Path(ra_dir) / f"ra_{sample_id:06d}.npy"

        if not all(p.exists() for p in [image_path, rd_path, ra_path]):
            continue

        # Create visualization
        fig = viz_tool.visualize_sample(
            sample_id, labels_df, predictions_df,
            image_path, rd_path, ra_path
        )

        # Save
        output_path = output_dir / f"comparison_{sample_id:06d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")


if __name__ == '__main__':
    # Example usage

    from pathlib import Path
    import argparse
    import os
    import json

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
    output_dir = ra_dir.parent / 'output_visualizations' / 'predictions_vs_labels'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    main(
        labels_csv=labels_csv,
        predictions_csv='./predictions/all_predictions.csv',
        image_dir=image_dir,
        rd_dir=rd_dir,
        ra_dir=ra_dir,
        output_dir=output_dir
    )