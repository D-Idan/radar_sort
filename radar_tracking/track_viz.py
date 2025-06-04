import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from radar_tracking import Detection, Track


def prepare_viz_directory(viz_dir: str):
    """
    Create viz_dir if it doesn't exist.
    Call once before the main for‐loop over frames begins.
    """
    os.makedirs(viz_dir, exist_ok=True)


def visualize_frame_radar_azimuth(
        frame_id: int,
        detections: List[Detection],
        ground_truth: List[Detection],
        active_tracks: List[Track],
        viz_dir: str
):
    """
    Plot one frame in (azimuth_deg, range_m) space:
      • detections: scatter as blue dots (Network Output)
      • ground_truth: scatter as green X's (Data Labels)
      • active_tracks: red circles with track IDs labeled

    Saves to: {viz_dir}/frame_{frame_id:06d}.jpg
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # 1) Plot network output (predicted detections)
    if detections:
        az_det = [np.degrees(d.azimuth_rad) for d in detections]
        rng_det = [d.range_m for d in detections]
        ax.scatter(az_det, rng_det, c='blue', s=20, label='Network Output', alpha=0.6)

    # 2) Plot data labels (ground truth)
    if ground_truth:
        az_gt = [np.degrees(d.azimuth_rad) for d in ground_truth]
        rng_gt = [d.range_m for d in ground_truth]
        ax.scatter(az_gt, rng_gt, c='green', marker='x', s=40, label='Data Labels')

    # 3) Plot active tracks
    for track in active_tracks:
        # Grab last_detection (assumed to exist)
        last_det = track.last_detection
        if last_det is None:
            continue
        az_tr = np.degrees(last_det.azimuth_rad)
        rng_tr = last_det.range_m
        ax.scatter(az_tr, rng_tr, c='red', s=30, edgecolors='black', linewidths=0.8)
        ax.text(az_tr + 0.2, rng_tr + 0.2, f"ID {track.id}", color='red', fontsize=8)

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"Frame {frame_id:06d}")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-45, 45)  # adjust as needed for your sensor's FoV
    ax.set_ylim(0, 120)  # adjust to your max radar range

    plt.tight_layout()
    out_path = os.path.join(viz_dir, f"frame_{frame_id:06d}.jpg")
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_counts_vs_tracks_per_frame(
        all_frames: List[int],
        det_counts: List[int],
        track_counts: List[int],
        path_save: str = "counts_vs_tracks_per_frame.png",
):
    """
    Plot Network Output per Frame vs. Confirmed Tracks per Frame
    """
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(all_frames, det_counts, marker='o', linestyle='-', label='Network Output/frame')
    ax1.plot(all_frames, track_counts, marker='s', linestyle='--', label='Active tracks/frame')
    ax1.set_xlabel("Frame ID (sample_id)")
    ax1.set_ylabel("Count")
    ax1.set_title("Network Output vs. Active Tracks per Frame")
    ax1.legend(loc='upper right')
    plt.tight_layout()
    fig1.savefig(path_save)
    plt.close(fig1)


def visualize_tracklet_lifetime_histogram(
        tracker,
        path_save: str = "tracklet_lifetime_histogram.png"
):
    """
    Generate histogram of tracklet lifetimes
    """
    # Get all stats from active and historical tracklets
    all_stats = {**tracker.active_tracklets, **tracker.historical_tracklets}
    lifetimes = np.array([sts.lifetime_frames for sts in all_stats.values()]) if all_stats else np.array([])

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if len(lifetimes) > 0:
        ax2.hist(lifetimes, bins=20, edgecolor='black')
    ax2.set_xlabel("Tracklet Lifetime (frames)")
    ax2.set_ylabel("Number of Tracklets")
    ax2.set_title("Histogram of Tracklet Lifetimes")
    plt.tight_layout()
    fig2.savefig(path_save)
    plt.close(fig2)


def visualize_avg_confidence_over_time(
        all_frames: List[int],
        avg_confidence_per_frame: List[float],
        path_save: str = "avg_confidence_over_time.png"
):
    """
    Plot average confidence of active tracks over time
    """
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(all_frames, avg_confidence_per_frame, marker='d', linestyle='-', color='tab:blue')
    ax3.set_xlabel("Frame ID (sample_id)")
    ax3.set_ylabel("Avg. Track Confidence")
    ax3.set_title("Average Confidence of Active Tracks Over Time")
    plt.tight_layout()
    fig3.savefig(path_save)
    plt.close(fig3)


def visualize_all_frames_overview(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        path_save: str = "all_frames_overview.png"
):
    """
    Create a comprehensive visualization showing all data from all frames:
    - Data Labels (ground truth) as green X's
    - Network Output (detections) as blue dots
    - Tracklets as connected red lines with different colors per track

    This visualization demonstrates the tracking benefits by showing how
    the tracker connects network outputs across frames on a moving platform.
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Color map for different track IDs
    colors = plt.cm.Set3(np.linspace(0, 1, 20))  # Generate 20 distinct colors
    track_colors = {}
    color_idx = 0

    # 1) Plot all data labels (ground truth) across all frames
    all_gt_az = []
    all_gt_rng = []
    for frame_idx, gt_list in enumerate(all_ground_truth):
        if gt_list:
            for gt in gt_list:
                all_gt_az.append(np.degrees(gt.azimuth_rad))
                all_gt_rng.append(gt.range_m)

    if all_gt_az:
        ax.scatter(all_gt_az, all_gt_rng, c='green', marker='x', s=30,
                   label='Data Labels', alpha=0.7, zorder=1)

    # 2) Plot all network outputs across all frames
    all_det_az = []
    all_det_rng = []
    for frame_idx, det_list in enumerate(all_detections):
        if det_list:
            for det in det_list:
                all_det_az.append(np.degrees(det.azimuth_rad))
                all_det_rng.append(det.range_m)

    if all_det_az:
        ax.scatter(all_det_az, all_det_rng, c='lightblue', s=15,
                   label='Network Output', alpha=0.5, zorder=2)

    # 3) Plot tracklets as connected paths
    # First, collect all track trajectories
    track_trajectories = {}

    for frame_idx, tracks in enumerate(all_tracks):
        for track in tracks:
            if track.last_detection is None:
                continue

            track_id = track.id
            if track_id not in track_trajectories:
                track_trajectories[track_id] = {
                    'azimuths': [],
                    'ranges': [],
                    'frames': []
                }

            az = np.degrees(track.last_detection.azimuth_rad)
            rng = track.last_detection.range_m

            track_trajectories[track_id]['azimuths'].append(az)
            track_trajectories[track_id]['ranges'].append(rng)
            track_trajectories[track_id]['frames'].append(all_frames[frame_idx])

    # Plot each track trajectory
    for track_id, trajectory in track_trajectories.items():
        if len(trajectory['azimuths']) < 2:  # Skip tracks with less than 2 points
            continue

        # Assign color to track
        if track_id not in track_colors:
            track_colors[track_id] = colors[color_idx % len(colors)]
            color_idx += 1

        color = track_colors[track_id]

        # Plot trajectory line
        ax.plot(trajectory['azimuths'], trajectory['ranges'],
                color=color, linewidth=2, alpha=0.8, zorder=3)

        # Plot track points
        ax.scatter(trajectory['azimuths'], trajectory['ranges'],
                   c=[color], s=40, edgecolors='black', linewidths=0.5, zorder=4)

        # Label the track at its last position
        if trajectory['azimuths']:
            last_az = trajectory['azimuths'][-1]
            last_rng = trajectory['ranges'][-1]
            ax.text(last_az + 0.5, last_rng + 0.5, f"T{track_id}",
                    color='red', fontsize=8, fontweight='bold', zorder=5)

    # Add legend entry for tracklets
    ax.plot([], [], color='red', linewidth=2, label='Tracklets', alpha=0.8)

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Radar Tracking Overview - All Frames\n" +
                 "Showing Tracker Benefits: Connecting Network Outputs Across Frames")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set reasonable limits (adjust based on your data range)
    ax.set_xlim(-45, 45)
    ax.set_ylim(0, 120)

    plt.tight_layout()
    plt.savefig(path_save, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_tracking_temporal_evolution(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        path_save: str = "tracking_temporal_evolution.png"
):
    """
    Create a temporal visualization showing how tracks evolve over time.
    This creates a multi-panel plot showing selected frames to demonstrate
    how the tracker maintains continuity across frames.
    """
    # Select a subset of frames to display (e.g., every 10th frame)
    display_frames_idx = list(range(0, len(all_frames), max(1, len(all_frames) // 6)))
    if len(display_frames_idx) > 6:
        display_frames_idx = display_frames_idx[:6]

    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    # Color map for track consistency across subplots
    colors = plt.cm.Set3(np.linspace(0, 1, 20))
    track_colors = {}
    color_idx = 0

    for plot_idx, frame_idx in enumerate(display_frames_idx):
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        frame_id = all_frames[frame_idx]

        # Plot network outputs for this frame
        detections = all_detections[frame_idx]
        if detections:
            az_det = [np.degrees(d.azimuth_rad) for d in detections]
            rng_det = [d.range_m for d in detections]
            ax.scatter(az_det, rng_det, c='lightblue', s=20, alpha=0.7)

        # Plot ground truth for this frame
        ground_truth = all_ground_truth[frame_idx]
        if ground_truth:
            az_gt = [np.degrees(d.azimuth_rad) for d in ground_truth]
            rng_gt = [d.range_m for d in ground_truth]
            ax.scatter(az_gt, rng_gt, c='green', marker='x', s=30)

        # Plot tracks for this frame
        tracks = all_tracks[frame_idx]
        for track in tracks:
            if track.last_detection is None:
                continue

            track_id = track.id
            if track_id not in track_colors:
                track_colors[track_id] = colors[color_idx % len(colors)]
                color_idx += 1

            color = track_colors[track_id]
            az = np.degrees(track.last_detection.azimuth_rad)
            rng = track.last_detection.range_m

            ax.scatter(az, rng, c=[color], s=50, edgecolors='black', linewidths=1)
            ax.text(az + 0.5, rng + 0.5, f"T{track_id}",
                    color='red', fontsize=8, fontweight='bold')

        ax.set_xlim(-45, 45)
        ax.set_ylim(0, 120)
        ax.set_title(f"Frame {frame_id}")
        ax.grid(True, alpha=0.3)

        if plot_idx >= cols * (rows - 1):  # Bottom row
            ax.set_xlabel("Azimuth (deg)")
        if plot_idx % cols == 0:  # Left column
            ax.set_ylabel("Range (m)")

    # Hide unused subplots
    for plot_idx in range(len(display_frames_idx), len(axes)):
        axes[plot_idx].set_visible(False)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
               markersize=8, label='Network Output'),
        Line2D([0], [0], marker='x', color='green', markersize=8,
               label='Data Labels', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Tracklets', markeredgecolor='black')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

    plt.suptitle("Tracking Evolution Across Selected Frames\n" +
                 "Demonstrating Track Continuity on Moving Platform", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(path_save, dpi=200, bbox_inches='tight')
    plt.close()