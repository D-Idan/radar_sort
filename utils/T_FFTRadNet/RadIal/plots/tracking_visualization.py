"""
Improved tracking visualization with consistent styling across all views.
Replaces the unsatisfactory BEV visualization with polar-map-style rendering.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib.patches as patches
from utils.T_FFTRadNet.RadIal.utils.util import worldToImage


class TrackingVisualizationTool:
    """Enhanced visualization tool for radar tracking results."""

    def __init__(self, camera_params=None, radar_params=None):
        # Default camera parameters
        self.camera_matrix = np.array([
            [1845.41929, 0.0, 855.802458],
            [0.0, 1788.69210, 607.342667],
            [0.0, 0.0, 1.0]
        ]) if camera_params is None else camera_params['matrix']

        self.image_width, self.image_height = 1920, 1080

        # Radar parameters
        self.radar_params = radar_params or {
            'max_range': 103.0,
            'min_azimuth': -90.0,
            'max_azimuth': 90.0
        }

        # Consistent styling across all visualizations
        self.style_config = {
            'ground_truth': {'color': 'green', 'marker': 'o', 'size': 80, 'alpha': 0.9, 'label': 'Ground Truth'},
            'detections': {'color': 'red', 'marker': 'x', 'size': 60, 'alpha': 0.8, 'label': 'Network Output'},
            'tracks': {'color': 'blue', 'marker': '^', 'size': 70, 'alpha': 0.9, 'label': 'Tracker State'},
            'coverage_bounds': {'color': 'gray', 'linestyle': '--', 'alpha': 0.5},
            'out_of_coverage': {'color': 'red', 'alpha': 0.1}
        }

    def load_data(self, labels_csv: str, predictions_csv: str, tracking_csv: Optional[str] = None):
        """Load all data sources."""
        labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')
        predictions_df = pd.read_csv(predictions_csv)
        tracking_df = pd.read_csv(tracking_csv) if tracking_csv is not None else None
        return labels_df, predictions_df, tracking_df

    def create_comprehensive_visualization(self, sample_id: int, labels_df: pd.DataFrame,
                                           predictions_df: pd.DataFrame, tracking_df: Optional[pd.DataFrame],
                                           image_path: Path, rd_path: Path, ra_path: Path) -> plt.Figure:
        """Create comprehensive 4-panel visualization with consistent styling."""

        # Load data files
        image = cv2.imread(str(image_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rd_map = np.load(rd_path)
        ra_map = np.load(ra_path)

        # Filter data for this sample
        sample_labels = labels_df[labels_df['numSample'] == sample_id]
        sample_predictions = predictions_df[predictions_df['sample_id'] == sample_id]
        sample_tracks = (tracking_df[tracking_df['sample_id'] == sample_id]
                         if tracking_df is not None else None)

        # Create figure with improved layout
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Radar Tracking Visualization - Sample {sample_id}', fontsize=16, fontweight='bold')

        # Panel 1: Camera image with bounding boxes
        if image is not None:
            image_viz = self._annotate_camera_image(image.copy(), sample_labels,
                                                    sample_predictions, sample_tracks)
            axs[0, 0].imshow(image_viz)
        axs[0, 0].set_title('Camera View with Projected Detections', fontweight='bold')
        axs[0, 0].axis('off')

        # Panel 2: Improved BEV with consistent styling
        self._create_bev_visualization(axs[0, 1], ra_map, sample_labels,
                                       sample_predictions, sample_tracks)
        axs[0, 1].set_title('Bird\'s Eye View (Consistent Styling)', fontweight='bold')

        # Panel 3: Range-Doppler map
        self._create_rd_visualization(axs[1, 0], rd_map)
        axs[1, 0].set_title('Range-Doppler Map', fontweight='bold')

        # Panel 4: Range-Azimuth map with annotations
        self._create_ra_visualization(axs[1, 1], ra_map, sample_labels,
                                      sample_predictions, sample_tracks)
        axs[1, 1].set_title('Range-Azimuth Map with Detections', fontweight='bold')

        # Add unified legend
        self._add_unified_legend(fig)

        plt.tight_layout()
        return fig

    def _annotate_camera_image(self, image: np.ndarray, labels_df: pd.DataFrame,
                               predictions_df: pd.DataFrame, tracks_df: Optional[pd.DataFrame]) -> np.ndarray:
        """Annotate camera image with bounding boxes."""

        def draw_bbox(img, bbox, label, color, thickness=2):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            # Add background for text readability
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            return img

        # Ground truth (green)
        for _, row in labels_df.iterrows():
            scale_w, scale_h = self._get_scale_factor(image)
            bbox = (row['x1_pix'] * scale_w, row['y1_pix'] * scale_h,
                    row['x2_pix'] * scale_w, row['y2_pix'] * scale_h)
            label = f"GT R:{row['radar_R_m']:.1f}m A:{row['radar_A_deg']:.1f}°"
            image = draw_bbox(image, bbox, label, (0, 255, 0), 3)

        # Network predictions (red)
        for _, row in predictions_df.iterrows():
            range_m, azimuth_deg = row['range_m'], row['azimuth_deg']
            x = np.sin(np.deg2rad(azimuth_deg)) * range_m
            y = np.cos(np.deg2rad(azimuth_deg)) * range_m

            u1, v1 = worldToImage(-x - 0.9, y, 0)
            u2, v2 = worldToImage(-x + 0.9, y, 1.6)
            u1, v1 = int(u1 / 2), int(v1 / 2)
            u2, v2 = int(u2 / 2), int(v2 / 2)

            bbox = (u1, v1, u2, v2)
            label = f"Det R:{range_m:.1f}m A:{azimuth_deg:.1f}° C:{row['confidence']:.2f}"
            image = draw_bbox(image, bbox, label, (255, 0, 0), 2)

        # Tracker predictions (blue)
        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                range_m, azimuth_deg = row['range_m'], row['azimuth_deg']
                x = np.sin(np.deg2rad(azimuth_deg)) * range_m
                y = np.cos(np.deg2rad(azimuth_deg)) * range_m

                u1, v1 = worldToImage(-x - 0.9, y, 0)
                u2, v2 = worldToImage(-x + 0.9, y, 1.6)
                u1, v1 = int(u1 / 2), int(v1 / 2)
                u2, v2 = int(u2 / 2), int(v2 / 2)

                bbox = (u1, v1, u2, v2)
                label = f"Track {row['track_id']} R:{range_m:.1f}m A:{azimuth_deg:.1f}° Age:{row['track_age']}"
                image = draw_bbox(image, bbox, label, (0, 0, 255), 2)

        return image

    def _create_bev_visualization(self, ax: plt.Axes, ra_map: np.ndarray,
                                  labels_df: pd.DataFrame, predictions_df: pd.DataFrame,
                                  tracks_df: Optional[pd.DataFrame]):
        """Create improved BEV visualization with consistent styling."""

        # Display the BEV background map
        bev_img = self._convert_ra_to_bev(ra_map)
        ax.imshow(bev_img, extent=[-103, 103, 0, 103], origin='lower', alpha=0.7, cmap='viridis')

        # Add radar coverage visualization
        self._add_coverage_bounds_bev(ax)

        # Convert and plot ground truth
        for _, row in labels_df.iterrows():
            x, y = self._polar_to_cartesian(row['radar_R_m'], row['radar_A_deg'])
            style = self.style_config['ground_truth']
            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size'],
                       alpha=style['alpha'], edgecolors='black', linewidth=1)

        # Convert and plot detections
        for _, row in predictions_df.iterrows():
            x, y = self._polar_to_cartesian(row['range_m'], row['azimuth_deg'])
            style = self.style_config['detections']
            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size'],
                       alpha=style['alpha'], linewidth=2)

        # Convert and plot tracks
        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                x, y = self._polar_to_cartesian(row['range_m'], row['azimuth_deg'])
                style = self.style_config['tracks']
                ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size'],
                           alpha=style['alpha'], edgecolors='white', linewidth=1)
                # Add track ID annotation
                ax.annotate(f"T{row['track_id']}", (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7))

        self._setup_bev_axes(ax)

    def _create_rd_visualization(self, ax: plt.Axes, rd_map: np.ndarray):
        """Create Range-Doppler visualization."""
        ax.imshow(rd_map, aspect='auto', origin='lower', cmap='jet')
        ax.invert_xaxis()
        self._setup_rd_axes(ax, rd_map.shape)

    def _create_ra_visualization(self, ax: plt.Axes, ra_map: np.ndarray,
                                 labels_df: pd.DataFrame, predictions_df: pd.DataFrame,
                                 tracks_df: Optional[pd.DataFrame]):
        """Create Range-Azimuth visualization with consistent styling."""

        # Display background map
        ax.imshow(ra_map, aspect='auto', origin='lower', alpha=0.8, cmap='viridis')
        ax.invert_xaxis()

        # Add coverage bounds
        self._add_coverage_bounds_ra(ax, ra_map.shape)

        # Plot ground truth
        for _, row in labels_df.iterrows():
            x, y = self._convert_ra_coords(row['radar_R_m'], row['radar_A_deg'], ra_map.shape)
            style = self.style_config['ground_truth']
            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size'],
                       alpha=style['alpha'], edgecolors='black', linewidth=1)

        # Plot detections
        for _, row in predictions_df.iterrows():
            x, y = self._convert_ra_coords(row['range_m'], row['azimuth_deg'], ra_map.shape)
            style = self.style_config['detections']
            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size'],
                       alpha=style['alpha'], linewidth=2)

        # Plot tracks
        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                x, y = self._convert_ra_coords(row['range_m'], row['azimuth_deg'], ra_map.shape)
                style = self.style_config['tracks']
                ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size'],
                           alpha=style['alpha'], edgecolors='white', linewidth=1)

        self._setup_ra_axes(ax, ra_map.shape)

    def _convert_ra_to_bev(self, ra_map: np.ndarray) -> np.ndarray:
        """Convert Range-Azimuth map to Bird's Eye View representation."""
        height, width = ra_map.shape
        bev_size = 512  # Fixed BEV image size
        bev_img = np.zeros((bev_size, bev_size))

        center = bev_size // 2
        max_range = self.radar_params['max_range']

        for i in range(height):
            for j in range(width):
                if ra_map[i, j] > 0:  # Only process non-zero pixels
                    # Convert RA coordinates to polar
                    range_m = (i / height) * max_range
                    azimuth_deg = ((j / width) * 180 - 90)

                    # Convert to BEV coordinates
                    x, y = self._polar_to_cartesian(range_m, azimuth_deg)

                    # Map to BEV image coordinates
                    x_pix = int(center + (x / max_range) * center)
                    y_pix = int(center - (y / max_range) * center)

                    if 0 <= x_pix < bev_size and 0 <= y_pix < bev_size:
                        bev_img[y_pix, x_pix] = ra_map[i, j]

        return bev_img

    def _polar_to_cartesian(self, range_m: float, azimuth_deg: float) -> Tuple[float, float]:
        """Convert polar coordinates to Cartesian."""
        azimuth_rad = np.deg2rad(azimuth_deg)
        x = range_m * np.sin(azimuth_rad)
        y = range_m * np.cos(azimuth_rad)
        return x, y

    def _convert_ra_coords(self, range_m: float, azimuth_deg: float, ra_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Convert range-azimuth to image coordinates."""
        height, width = ra_shape
        y = np.clip(int((range_m / self.radar_params['max_range']) * height), 0, height - 1)
        x = np.clip(int(((azimuth_deg + 90) / 180) * width), 0, width - 1)
        return x, y

    def _add_coverage_bounds_bev(self, ax: plt.Axes):
        """Add radar coverage bounds to BEV plot."""
        max_range = self.radar_params['max_range']

        # Draw range circle
        circle = plt.Circle((0, 0), max_range, fill=False,
                            color=self.style_config['coverage_bounds']['color'],
                            linestyle=self.style_config['coverage_bounds']['linestyle'],
                            alpha=self.style_config['coverage_bounds']['alpha'])
        ax.add_patch(circle)

        # Draw azimuth sector
        min_az = self.radar_params['min_azimuth']
        max_az = self.radar_params['max_azimuth']

        # Left bound
        x1 = max_range * np.sin(np.deg2rad(min_az))
        y1 = max_range * np.cos(np.deg2rad(min_az))
        ax.plot([0, x1], [0, y1], color=self.style_config['coverage_bounds']['color'],
                linestyle=self.style_config['coverage_bounds']['linestyle'],
                alpha=self.style_config['coverage_bounds']['alpha'])

        # Right bound
        x2 = max_range * np.sin(np.deg2rad(max_az))
        y2 = max_range * np.cos(np.deg2rad(max_az))
        ax.plot([0, x2], [0, y2], color=self.style_config['coverage_bounds']['color'],
                linestyle=self.style_config['coverage_bounds']['linestyle'],
                alpha=self.style_config['coverage_bounds']['alpha'])

    def _add_coverage_bounds_ra(self, ax: plt.Axes, ra_shape: Tuple[int, int]):
        """Add radar coverage bounds to RA plot."""
        height, width = ra_shape

        # Max range line
        max_range_y = (self.radar_params['max_range'] / self.radar_params['max_range']) * height
        style = self.style_config['coverage_bounds']
        ax.axhline(y=max_range_y, color=style['color'], linestyle=style['linestyle'], alpha=style['alpha'])

        # Azimuth bounds
        min_az_x = ((self.radar_params['min_azimuth'] + 90) / 180) * width
        max_az_x = ((self.radar_params['max_azimuth'] + 90) / 180) * width
        ax.axvline(x=min_az_x, color=style['color'], linestyle=style['linestyle'], alpha=style['alpha'])
        ax.axvline(x=max_az_x, color=style['color'], linestyle=style['linestyle'], alpha=style['alpha'])

    def _setup_bev_axes(self, ax: plt.Axes):
        """Setup BEV axes with proper labels."""
        max_range = self.radar_params['max_range']
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(0, max_range)
        ax.set_xlabel('Lateral Distance (m)', fontweight='bold')
        ax.set_ylabel('Forward Distance (m)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def _setup_rd_axes(self, ax: plt.Axes, shape: Tuple[int, int]):
        """Setup Range-Doppler axes."""
        height, width = shape

        # Doppler axis
        doppler_ticks = np.linspace(-width // 2 * 0.1, width // 2 * 0.1, 5)
        x_ticks = doppler_ticks / 0.1 + width / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in doppler_ticks])
        ax.set_xlabel("Doppler Velocity (m/s)", fontweight='bold')

        # Range axis
        range_ticks = [0, 20, 40, 60, 80, 100]
        y_ticks = [(r / self.radar_params['max_range']) * height for r in range_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{r}m" for r in range_ticks])
        ax.set_ylabel("Range (m)", fontweight='bold')

    def _setup_ra_axes(self, ax: plt.Axes, shape: Tuple[int, int]):
        """Setup Range-Azimuth axes."""
        height, width = shape

        # Azimuth axis
        azimuth_ticks = [-90, -45, 0, 45, 90]
        x_ticks = [((a + 90) / 180) * width for a in azimuth_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{a}°" for a in azimuth_ticks])
        ax.set_xlabel("Azimuth Angle (degrees)", fontweight='bold')

        # Range axis
        range_ticks = [0, 20, 40, 60, 80, 100]
        y_ticks = [(r / self.radar_params['max_range']) * height for r in range_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{r}m" for r in range_ticks])
        ax.set_ylabel("Range (m)", fontweight='bold')

    def _add_unified_legend(self, fig: plt.Figure):
        """Add unified legend for all plots."""
        legend_elements = []
        for key, style in self.style_config.items():
            if 'label' in style:
                legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], color='w',
                                                  markerfacecolor=style['color'], markersize=8,
                                                  alpha=style['alpha'], label=style['label']))

        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                   ncol=len(legend_elements), fontsize=12, frameon=True, fancybox=True, shadow=True)

    def _get_scale_factor(self, image: np.ndarray) -> Tuple[float, float]:
        """Calculate scale factors for image."""
        height, width = image.shape[:2]
        return width / self.image_width, height / self.image_height


def create_tracking_video(data_dir: Path, output_dir: Path, labels_csv: str,
                          predictions_csv: str, tracking_csv: Optional[str] = None,
                          max_samples: Optional[int] = None) -> str:
    """Create video from tracking visualizations."""

    viz_tool = TrackingVisualizationTool()
    labels_df, predictions_df, tracking_df = viz_tool.load_data(labels_csv, predictions_csv, tracking_csv)

    # Create temporary directory for frames
    frames_dir = output_dir / 'video_frames'
    frames_dir.mkdir(exist_ok=True)

    # Get sample IDs
    sample_ids = sorted(labels_df['numSample'].unique())
    if max_samples:
        sample_ids = sample_ids[:max_samples]

    created_frames = []

    for i, sample_id in enumerate(sample_ids):
        # File paths
        image_path = data_dir / 'camera' / f"image_{sample_id:06d}.jpg"
        rd_path = data_dir / 'radar_RD' / f"rd_{sample_id:06d}.npy"
        ra_path = data_dir / 'radar_RA' / f"ra_{sample_id:06d}.npy"

        if not all(p.exists() for p in [image_path, rd_path, ra_path]):
            continue

        # Create visualization
        fig = viz_tool.create_comprehensive_visualization(
            sample_id, labels_df, predictions_df, tracking_df,
            image_path, rd_path, ra_path
        )

        # Save frame
        frame_path = frames_dir / f"frame_{i:06d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        created_frames.append(frame_path)

        if i % 10 == 0:
            print(f"Created frame {i + 1}/{len(sample_ids)}")

    # Create video using ffmpeg (if available)
    video_path = output_dir / 'tracking_visualization.mp4'
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-framerate', '5',
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(video_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video created: {video_path}")

        # Clean up frames
        for frame in created_frames:
            frame.unlink()
        frames_dir.rmdir()

    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"FFmpeg not available. Frame images saved in: {frames_dir}")
        video_path = frames_dir

    return str(video_path)