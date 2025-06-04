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

    # Plot network output (blue circles)
    if detections:
        az_det = [np.degrees(d.azimuth_rad) for d in detections]
        rng_det = [d.range_m for d in detections]
        ax.scatter(az_det, rng_det, c='blue', s=20, label='Network Output', alpha=1.0)

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
    ax.set_title(f"Frame {frame_id:06d}")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(90, -90)  # adjust as needed for your sensor's FoV
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


def visualize_all_frames_3d_overview(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        path_save: str = "3d_tracking_overview.png"
):
    """Create 3D visualization with time axis to show motion on moving platform."""
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
            # Start point
            ax.text(
                data['times'][0],
                data['azimuths'][0],
                data['ranges'][0],
                f"T{track_id}",
                color=color,
                fontsize=8
            )
            # End point
            ax.text(
                data['times'][-1],
                data['azimuths'][-1],
                data['ranges'][-1],
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
    # Only show 1 entry per track ID
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

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
    # Find longest-lived tracks
    track_lifespans = {}
    for frame_idx, tracks in enumerate(all_tracks):
        for track in tracks:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((all_frames[frame_idx], track))

    # Select top 3 longest tracks
    longest_tracks = sorted(track_lifespans.items(), key=lambda x: len(x[1]), reverse=True)[:3]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    colors = ['red', 'blue', 'orange']

    for i, (track_id, track_history) in enumerate(longest_tracks):
        ax = axes[i]

        frames = [frame for frame, _ in track_history]
        ranges = [track.kalman_polar_position[0] for _, track in track_history]

        # Plot track trajectory (smoothed)
        ax.plot(frames, ranges, color=colors[i], linewidth=2, label=f'Track {track_id} (Smoothed)')

        # Plot raw detections for comparison
        for frame, track in track_history:
            if track.last_detection:
                ax.scatter(frame, track.last_detection.range_m,
                           color=colors[i], alpha=0.3, s=20, marker='o')

        ax.set_ylabel('Range (m)')
        ax.set_title(f'Track {track_id}: Kalman Smoothing vs Raw Detections')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame ID')
    plt.suptitle('Temporal Evolution: Showing Tracker Benefits\n(Smoothing, Gap-filling, False Association Avoidance)')
    plt.tight_layout()
    plt.savefig(path_save, dpi=200, bbox_inches='tight')
    plt.close()