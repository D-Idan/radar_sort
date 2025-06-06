import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional
from radar_tracking import Detection, Track


def prepare_output_directories(output_dir: str):
    """
    Create output directory structure for visualizations.

    Args:
        output_dir: Base output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def visualize_frame_radar_azimuth(
        frame_id: int,
        detections: List[Detection],
        ground_truth: List[Detection],
        active_tracks: List[Track],
        output_dir: str,
        show_coverage_bounds: bool = True
):
    """
    Plot one frame in (azimuth_deg, range_m) space with radar coverage overlay.
    """
    prepare_output_directories(output_dir)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Show radar coverage bounds
    if show_coverage_bounds:
        # Draw coverage area
        ax.axhline(y=103, color='gray', linestyle='--', alpha=0.5, label='Max Range')
        ax.axvline(x=-90, color='gray', linestyle='--', alpha=0.5, label='Azimuth Limits')
        ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5)

        # Shade out-of-coverage areas
        ax.fill_between([-90, 90], 103, 120, color='red', alpha=0.1, label='Out of Coverage')
        ax.fill([-100, -90, -90, -100], [0, 0, 120, 120], color='red', alpha=0.1)
        ax.fill([90, 100, 100, 90], [0, 0, 120, 120], color='red', alpha=0.1)

    # Plot network output (blue circles)
    if detections:
        az_det = [np.degrees(d.azimuth_rad) for d in detections]
        rng_det = [d.range_m for d in detections]
        conf_det = [d.confidence for d in detections]

        # Color by confidence
        scatter = ax.scatter(az_det, rng_det, c=conf_det, s=20,
                             cmap='Blues', alpha=0.8, vmin=0, vmax=1,
                             label='Network Output')
        plt.colorbar(scatter, ax=ax, label='Confidence')

    # Plot ground truth (green X)
    if ground_truth:
        az_gt = [np.degrees(d.azimuth_rad) for d in ground_truth]
        rng_gt = [d.range_m for d in ground_truth]
        ax.scatter(az_gt, rng_gt, c='green', marker='x', s=60, label='Ground Truth')

    # Plot tracks using Kalman state (red triangles)
    for track in active_tracks:
        range_m, azimuth_rad = track.kalman_polar_position
        az_tr = np.degrees(azimuth_rad)
        rng_tr = range_m

        ax.scatter(az_tr, rng_tr, marker='^', s=20, facecolors='none', edgecolors='red',
                   linewidths=0.8, label='Tracker Estimate' if track == active_tracks[0] else "")
        ax.text(az_tr + 0.2, rng_tr + 0.2, f"T{track.id}", color='red', fontsize=8)

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"Frame {frame_id:06d} - Radar Coverage Visualization")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-100, 100)  # Show slightly beyond coverage for context
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_counts_vs_tracks_per_frame(
        all_frames: List[int],
        det_counts: List[int],
        track_counts: List[int],
        output_dir: str
):
    """Plot Network Output per Frame vs. Confirmed Tracks per Frame."""
    prepare_output_directories(output_dir)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(all_frames, det_counts, marker='o', linestyle='-', label='Network Output/frame')
    ax1.plot(all_frames, track_counts, marker='s', linestyle='--', label='Active tracks/frame')
    ax1.set_xlabel("Frame ID (sample_id)")
    ax1.set_ylabel("Count")
    ax1.set_title("Network Output vs. Active Tracks per Frame")
    ax1.legend(loc='upper right')
    plt.tight_layout()

    save_path = Path(output_dir) / "counts_vs_tracks_per_frame.png"
    fig1.savefig(save_path)
    plt.close(fig1)


def visualize_tracklet_lifetime_histogram(
        manager,
        output_dir: str
):
    """Generate histogram of tracklet lifetimes."""
    prepare_output_directories(output_dir)

    # Get all stats from active and historical tracklets
    all_stats = {**manager.active_tracklets, **manager.historical_tracklets}
    lifetimes = np.array([sts.lifetime_frames for sts in all_stats.values()]) if all_stats else np.array([])

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if len(lifetimes) > 0:
        ax2.hist(lifetimes, bins=20, edgecolor='black')
    ax2.set_xlabel("Tracklet Lifetime (frames)")
    ax2.set_ylabel("Number of Tracklets")
    ax2.set_title("Histogram of Tracklet Lifetimes")
    plt.tight_layout()

    save_path = Path(output_dir) / "tracklet_lifetime_histogram.png"
    fig2.savefig(save_path)
    plt.close(fig2)


def visualize_avg_confidence_over_time(
        all_frames: List[int],
        avg_confidence_per_frame: List[float],
        output_dir: str
):
    """Plot average confidence of active tracks over time."""
    prepare_output_directories(output_dir)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(all_frames, avg_confidence_per_frame, marker='d', linestyle='-', color='tab:blue')
    ax3.set_xlabel("Frame ID (sample_id)")
    ax3.set_ylabel("Avg. Track Confidence")
    ax3.set_title("Average Confidence of Active Tracks Over Time")
    plt.tight_layout()

    save_path = Path(output_dir) / "avg_confidence_over_time.png"
    fig3.savefig(save_path)
    plt.close(fig3)


def visualize_all_frames_3d_overview(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        output_dir: str
):
    """Create 3D visualization with time axis."""
    prepare_output_directories(output_dir)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Collect all data points
    gt_times, gt_az, gt_rng = [], [], []
    det_times, det_az, det_rng = [], [], []

    # Create a color map for tracks
    unique_track_ids = set()
    for frame_tracks in all_tracks:
        for track in frame_tracks:
            unique_track_ids.add(track.id)
    num_tracks = len(unique_track_ids)
    color_map = cm.get_cmap('hsv', num_tracks + 1)
    track_colors = {track_id: mcolors.to_hex(color_map(i))
                    for i, track_id in enumerate(unique_track_ids)}

    # Store track data by ID
    track_data = {}
    for frame_idx, frame_id in enumerate(all_frames):
        # Ground truth
        for gt in all_ground_truth[frame_idx]:
            gt_times.append(frame_id)
            gt_az.append(np.degrees(gt.azimuth_rad))
            gt_rng.append(gt.range_m)

        # Detections
        for det in all_detections[frame_idx]:
            det_times.append(frame_id)
            det_az.append(np.degrees(det.azimuth_rad))
            det_rng.append(det.range_m)

        # Tracks
        for track in all_tracks[frame_idx]:
            range_m, azimuth_rad = track.kalman_polar_position
            azimuth_deg = np.degrees(azimuth_rad)

            if track.id not in track_data:
                track_data[track.id] = {
                    'times': [],
                    'azimuths': [],
                    'ranges': []
                }

            track_data[track.id]['times'].append(frame_id)
            track_data[track.id]['azimuths'].append(azimuth_deg)
            track_data[track.id]['ranges'].append(range_m)

    # Plot 3D scatter
    ax.scatter(gt_times, gt_az, gt_rng, c='green', marker='x', s=40, alpha=0.7, label='Ground Truth')
    ax.scatter(det_times, det_az, det_rng, c='blue', s=15, alpha=0.5, label='Detections')

    # Plot tracks with unique colors and connecting lines
    for track_id, data in track_data.items():
        color = track_colors[track_id]
        ax.plot(
            data['times'],
            data['azimuths'],
            data['ranges'],
            color=color,
            marker='',
            markersize=4,
            linewidth=1.5,
            alpha=0.7,
            label=f'Track {track_id}'
        )

        # Add track ID labels at start and end points
        if data['times']:
            ax.text(
                data['times'][0],
                data['azimuths'][0],
                data['ranges'][0],
                f"T{track_id}",
                color=color,
                fontsize=8
            )

    ax.set_xlabel('Frame ID (Time)')
    ax.set_ylabel('Azimuth (deg)')
    ax.set_zlabel('Range (m)')
    ax.set_title('3D Radar Tracking: Space + Time View\nTracks Shown with Unique Colors')

    # Create a legend with track colors
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

    plt.tight_layout()
    save_path = Path(output_dir) / "3d_tracking_overview.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_tracking_temporal_evolution(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        num_tracks: Optional[int] = 3,
        output_dir: str = "tracking_temporal_evolution.png",
):
    """Create a temporal visualization showing how tracks evolve over time."""
    prepare_output_directories(output_dir)

    # Find longest-lived tracks
    track_lifespans = {}
    for frame_idx, tracks in enumerate(all_tracks):
        for track in tracks:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((all_frames[frame_idx], track))

    # Select top num_tracks longest tracks
    longest_tracks = sorted(track_lifespans.items(), key=lambda x: len(x[1]), reverse=True)[:num_tracks]

    fig, axes = plt.subplots(num_tracks, 1, figsize=(12, 10))
    if num_tracks == 1:
        axes = [axes]  # Make it iterable

    colors = ['red', 'blue', 'orange']

    for i, (track_id, track_history) in enumerate(longest_tracks):
        if i >= len(axes):
            break

        ax = axes[i]

        frames = [frame for frame, _ in track_history]
        ranges = [track.kalman_polar_position[0] for _, track in track_history]

        # Plot track trajectory (smoothed)
        ax.plot(frames, ranges, color=colors[i % len(colors)], linewidth=2, label=f'Track {track_id} (Smoothed)')

        # Plot raw detections for comparison
        for frame, track in track_history:
            if track.last_detection:
                ax.scatter(frame, track.last_detection.range_m,
                           color=colors[i % len(colors)], alpha=0.3, s=20, marker='o')

        ax.set_ylabel('Range (m)')
        ax.set_title(f'Track {track_id}: Kalman Smoothing vs Raw Detections')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(axes) > 0:
        axes[-1].set_xlabel('Frame ID')
    plt.suptitle('Temporal Evolution: Showing Tracker Benefits\n(Smoothing, Gap-filling, False Association Avoidance)')
    plt.tight_layout()

    save_path = Path(output_dir) / "tracking_temporal_evolution.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()