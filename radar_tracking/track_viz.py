import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
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
        all_tracks: List[List[Track]],
        frame_times: List[Tuple[int, float]],
        output_dir: str,
        window_size: int = 3
):
    """Plot confidence over time for each track with rolling window averaging."""
    prepare_output_directories(output_dir)

    # Create frame to timestamp mapping
    frame_to_time = dict(frame_times)

    # Collect confidence data per track
    track_confidence_data = {}

    for frame_idx, (frame_id, tracks) in enumerate(zip(all_frames, all_tracks)):
        timestamp = frame_to_time.get(frame_id, frame_id)

        for track in tracks:
            if track.id not in track_confidence_data:
                track_confidence_data[track.id] = {
                    'times': [],
                    'confidences': [],
                    'raw_confidences': []
                }

            if track.last_detection and track.last_detection.confidence > 0:
                track_confidence_data[track.id]['times'].append(timestamp)
                track_confidence_data[track.id]['raw_confidences'].append(
                    track.last_detection.confidence
                )

    # Apply rolling window averaging
    for track_id, data in track_confidence_data.items():
        raw_conf = data['raw_confidences']
        smoothed_conf = []

        for i in range(len(raw_conf)):
            # Get window of last 'window_size' values
            start_idx = max(0, i - window_size + 1)
            window = raw_conf[start_idx:i + 1]
            smoothed_conf.append(np.mean(window))

        data['confidences'] = smoothed_conf

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Main plot: Individual track confidence
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Sort tracks by lifetime for better visualization
    sorted_tracks = sorted(track_confidence_data.items(),
                           key=lambda x: len(x[1]['times']), reverse=True)

    # Plot top N tracks
    max_tracks_to_show = 10
    shown_tracks = 0

    for i, (track_id, data) in enumerate(sorted_tracks):
        if shown_tracks >= max_tracks_to_show:
            break

        if len(data['times']) < 5:  # Skip very short tracks
            continue

        color = colors[shown_tracks % len(colors)]

        # Plot smoothed confidence
        ax1.plot(data['times'], data['confidences'],
                 color=color, linewidth=2.5, alpha=0.8,
                 label=f'Track {track_id} (smoothed)', zorder=2)

        # Plot raw confidence as scatter
        ax1.scatter(data['times'], data['raw_confidences'],
                    color=color, s=20, alpha=0.3, zorder=1)

        shown_tracks += 1

    # Add overall average
    timestamps = [frame_to_time.get(f, f) for f in all_frames]
    ax1.plot(timestamps, avg_confidence_per_frame,
             'k--', linewidth=2, alpha=0.7,
             label='Overall Average', zorder=3)

    ax1.set_ylabel('Detection Confidence', fontsize=12)
    ax1.set_title(f'Track Confidence Over Time (Rolling Window Size: {window_size})\n'
                  f'Showing top {shown_tracks} longest tracks', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Bottom plot: Track count over time
    track_counts = []
    for tracks in all_tracks:
        track_counts.append(len([t for t in tracks if t.last_detection]))

    ax2.fill_between(timestamps, track_counts, alpha=0.3, color='gray')
    ax2.plot(timestamps, track_counts, color='darkgray', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Active Tracks', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / "avg_confidence_over_time.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_all_frames_3d_overview(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],  # Added parameter
        output_dir: str
):
    """Create 3D visualization with real time axis showing tracker predictions during gaps."""
    prepare_output_directories(output_dir)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create frame_id to timestamp mapping
    frame_to_time = dict(frame_times)

    # Collect all data points
    gt_times, gt_az, gt_rng = [], [], []
    det_times, det_az, det_rng = [], [], []

    # Create a color map for tracks
    unique_track_ids = set()
    for frame_tracks in all_tracks:
        for track in frame_tracks:
            unique_track_ids.add(track.id)
    num_tracks = len(unique_track_ids)
    color_map = cm.get_cmap('tab10', min(num_tracks + 1, 10))
    track_colors = {track_id: mcolors.to_hex(color_map(i % 10))
                    for i, track_id in enumerate(unique_track_ids)}

    # Store track data by ID with timestamps
    track_data = {}

    for frame_idx, frame_id in enumerate(all_frames):
        timestamp = frame_to_time.get(frame_id, frame_id)

        # Ground truth
        for gt in all_ground_truth[frame_idx]:
            gt_times.append(timestamp)
            gt_az.append(np.degrees(gt.azimuth_rad))
            gt_rng.append(gt.range_m)

        # Detections
        for det in all_detections[frame_idx]:
            det_times.append(timestamp)
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
                    'ranges': [],
                    'has_detection': []
                }

            track_data[track.id]['times'].append(timestamp)
            track_data[track.id]['azimuths'].append(azimuth_deg)
            track_data[track.id]['ranges'].append(range_m)
            track_data[track.id]['has_detection'].append(track.last_detection is not None)

    # Plot ground truth and detections
    if gt_times:
        ax.scatter(gt_times, gt_az, gt_rng, c='green', marker='x', s=40,
                   alpha=0.7, label='Ground Truth')
    if det_times:
        ax.scatter(det_times, det_az, det_rng, c='blue', s=15,
                   alpha=0.5, label='Detections')

    # Plot tracks with predictions during gaps
    for track_id, data in track_data.items():
        if not data['times']:
            continue

        color = track_colors[track_id]
        times = data['times']
        azimuths = data['azimuths']
        ranges = data['ranges']
        has_detection = data['has_detection']

        # Interpolate predictions during time gaps
        interpolated_times = []
        interpolated_az = []
        interpolated_rng = []

        for i in range(len(times) - 1):
            interpolated_times.append(times[i])
            interpolated_az.append(azimuths[i])
            interpolated_rng.append(ranges[i])

            # Check for time gap
            time_gap = times[i + 1] - times[i]
            if time_gap > 0.2:  # If gap > 200ms, show predictions
                # Add interpolated points
                num_points = int(time_gap / 0.1)  # One point every 100ms
                for j in range(1, num_points):
                    t = times[i] + j * 0.1
                    # Linear interpolation for visualization
                    alpha = j / num_points
                    az = azimuths[i] + alpha * (azimuths[i + 1] - azimuths[i])
                    rng = ranges[i] + alpha * (ranges[i + 1] - ranges[i])
                    interpolated_times.append(t)
                    interpolated_az.append(az)
                    interpolated_rng.append(rng)

        # Add last point
        interpolated_times.append(times[-1])
        interpolated_az.append(azimuths[-1])
        interpolated_rng.append(ranges[-1])

        # Plot track with different styles for measurements vs predictions
        ax.plot(interpolated_times, interpolated_az, interpolated_rng,
                color=color, linewidth=2, alpha=0.8, label=f'Track {track_id}')

        # Highlight actual measurements
        measurement_times = [t for t, has_det in zip(times, has_detection) if has_det]
        measurement_az = [az for az, has_det in zip(azimuths, has_detection) if has_det]
        measurement_rng = [r for r, has_det in zip(ranges, has_detection) if has_det]

        if measurement_times:
            ax.scatter(measurement_times, measurement_az, measurement_rng,
                       color=color, s=30, marker='o', edgecolors='black',
                       linewidth=0.5, alpha=1.0)

        # Add track ID at start
        if times:
            ax.text(times[0], azimuths[0], ranges[0], f"T{track_id}",
                    color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Azimuth (degrees)', fontsize=12)
    ax.set_zlabel('Range (meters)', fontsize=12)
    ax.set_title('3D Radar Tracking: Real-Time View with Predictions During Gaps\n'
                 'Solid lines show continuous tracking, dots show actual measurements',
                 fontsize=14)

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    save_path = Path(output_dir) / "3d_tracking_overview.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_tracking_temporal_evolution(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],  # Added parameter
        num_tracks: Optional[int] = 3,
        output_dir: str = "tracking_temporal_evolution.png",
):
    """Create temporal visualization with real time axis showing predictions during gaps."""
    prepare_output_directories(output_dir)

    # Create frame_id to timestamp mapping
    frame_to_time = dict(frame_times)

    # Find longest-lived tracks
    track_lifespans = {}
    for frame_idx, tracks in enumerate(all_tracks):
        timestamp = frame_to_time.get(all_frames[frame_idx], all_frames[frame_idx])
        for track in tracks:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((timestamp, track, all_frames[frame_idx]))

    # Select top num_tracks longest tracks
    longest_tracks = sorted(track_lifespans.items(), key=lambda x: len(x[1]), reverse=True)[:num_tracks]

    fig, axes = plt.subplots(num_tracks, 1, figsize=(14, 3.5 * num_tracks))
    if num_tracks == 1:
        axes = [axes]

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, (track_id, track_history) in enumerate(longest_tracks):
        if i >= len(axes):
            break

        ax = axes[i]

        # Extract data
        timestamps = [t for t, _, _ in track_history]
        ranges = [track.kalman_polar_position[0] for _, track, _ in track_history]
        has_detection = [track.last_detection is not None for _, track, _ in track_history]

        # Create dense time series for predictions
        dense_times = []
        dense_ranges = []
        prediction_mask = []

        for j in range(len(timestamps) - 1):
            dense_times.append(timestamps[j])
            dense_ranges.append(ranges[j])
            prediction_mask.append(True)  # Actual measurement

            # Fill gaps with predictions
            time_gap = timestamps[j + 1] - timestamps[j]
            if time_gap > 0.15:  # Show predictions for gaps > 150ms
                num_pred = int(time_gap / 0.05)  # Prediction every 50ms
                for k in range(1, num_pred):
                    t = timestamps[j] + k * 0.05
                    # Linear interpolation for smooth visualization
                    alpha = k / num_pred
                    r = ranges[j] + alpha * (ranges[j + 1] - ranges[j])
                    dense_times.append(t)
                    dense_ranges.append(r)
                    prediction_mask.append(False)  # Prediction

        # Add last point
        dense_times.append(timestamps[-1])
        dense_ranges.append(ranges[-1])
        prediction_mask.append(True)

        # Plot Kalman filtered trajectory
        ax.plot(dense_times, dense_ranges, color=colors[i % len(colors)],
                linewidth=2.5, label=f'Track {track_id} (Kalman Filtered)', zorder=2)

        # Highlight predictions with different style
        pred_times = [t for t, is_meas in zip(dense_times, prediction_mask) if not is_meas]
        pred_ranges = [r for r, is_meas in zip(dense_ranges, prediction_mask) if not is_meas]
        if pred_times:
            ax.plot(pred_times, pred_ranges, 'o', color=colors[i % len(colors)],
                    markersize=3, alpha=0.3, label='Predictions during gaps')

        # Plot raw detections
        detection_times = []
        detection_ranges = []
        for timestamp, track, _ in track_history:
            if track.last_detection:
                detection_times.append(timestamp)
                detection_ranges.append(track.last_detection.range_m)

        if detection_times:
            ax.scatter(detection_times, detection_ranges,
                       color=colors[i % len(colors)], alpha=0.6, s=40,
                       marker='o', edgecolors='black', linewidth=0.5,
                       label='Raw Detections', zorder=3)

        # Mark time gaps
        for j in range(len(timestamps) - 1):
            gap = timestamps[j + 1] - timestamps[j]
            if gap > 0.5:  # Mark gaps > 500ms
                ax.axvspan(timestamps[j], timestamps[j + 1], alpha=0.1,
                           color='red', label='Time gap' if j == 0 else "")
                # Add text annotation
                mid_time = (timestamps[j] + timestamps[j + 1]) / 2
                ax.text(mid_time, ax.get_ylim()[1] * 0.95, f'{gap:.1f}s gap',
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_ylabel('Range (m)', fontsize=11)
        ax.set_title(f'Track {track_id}: Continuous Tracking Through Time Gaps', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set x-axis to show actual time
        ax.set_xlim(min(dense_times) - 0.5, max(dense_times) + 0.5)

    if len(axes) > 0:
        axes[-1].set_xlabel('Time (seconds)', fontsize=11)

    plt.suptitle('Temporal Evolution: Real-Time Tracking with Predictions\n'
                 'Tracker maintains position estimates during measurement gaps',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = Path(output_dir) / "tracking_temporal_evolution.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_tracking_temporal_evolution(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],  # Added parameter
        num_tracks: Optional[int] = 3,
        output_dir: str = "tracking_temporal_evolution.png",
):
    """Create temporal visualization with real time axis showing predictions during gaps."""
    prepare_output_directories(output_dir)

    # Create frame_id to timestamp mapping
    frame_to_time = dict(frame_times)

    # Find longest-lived tracks
    track_lifespans = {}
    for frame_idx, tracks in enumerate(all_tracks):
        timestamp = frame_to_time.get(all_frames[frame_idx], all_frames[frame_idx])
        for track in tracks:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((timestamp, track, all_frames[frame_idx]))

    # Select top num_tracks longest tracks
    longest_tracks = sorted(track_lifespans.items(), key=lambda x: len(x[1]), reverse=True)[:num_tracks]

    fig, axes = plt.subplots(num_tracks, 1, figsize=(14, 3.5 * num_tracks))
    if num_tracks == 1:
        axes = [axes]

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, (track_id, track_history) in enumerate(longest_tracks):
        if i >= len(axes):
            break

        ax = axes[i]

        # Extract data
        timestamps = [t for t, _, _ in track_history]
        ranges = [track.kalman_polar_position[0] for _, track, _ in track_history]
        has_detection = [track.last_detection is not None for _, track, _ in track_history]

        # Create dense time series for predictions
        dense_times = []
        dense_ranges = []
        prediction_mask = []

        for j in range(len(timestamps) - 1):
            dense_times.append(timestamps[j])
            dense_ranges.append(ranges[j])
            prediction_mask.append(True)  # Actual measurement

            # Fill gaps with predictions
            time_gap = timestamps[j + 1] - timestamps[j]
            if time_gap > 0.15:  # Show predictions for gaps > 150ms
                num_pred = int(time_gap / 0.05)  # Prediction every 50ms
                for k in range(1, num_pred):
                    t = timestamps[j] + k * 0.05
                    # Linear interpolation for smooth visualization
                    alpha = k / num_pred
                    r = ranges[j] + alpha * (ranges[j + 1] - ranges[j])
                    dense_times.append(t)
                    dense_ranges.append(r)
                    prediction_mask.append(False)  # Prediction

        # Add last point
        dense_times.append(timestamps[-1])
        dense_ranges.append(ranges[-1])
        prediction_mask.append(True)

        # Plot Kalman filtered trajectory
        ax.plot(dense_times, dense_ranges, color=colors[i % len(colors)],
                linewidth=2.5, label=f'Track {track_id} (Kalman Filtered)', zorder=2)

        # Highlight predictions with different style
        pred_times = [t for t, is_meas in zip(dense_times, prediction_mask) if not is_meas]
        pred_ranges = [r for r, is_meas in zip(dense_ranges, prediction_mask) if not is_meas]
        if pred_times:
            ax.plot(pred_times, pred_ranges, 'o', color=colors[i % len(colors)],
                    markersize=3, alpha=0.3, label='Predictions during gaps')

        # Plot raw detections
        detection_times = []
        detection_ranges = []
        for timestamp, track, _ in track_history:
            if track.last_detection:
                detection_times.append(timestamp)
                detection_ranges.append(track.last_detection.range_m)

        if detection_times:
            ax.scatter(detection_times, detection_ranges,
                       color=colors[i % len(colors)], alpha=0.6, s=40,
                       marker='o', edgecolors='black', linewidth=0.5,
                       label='Raw Detections', zorder=3)

        # Mark time gaps
        for j in range(len(timestamps) - 1):
            gap = timestamps[j + 1] - timestamps[j]
            if gap > 0.5:  # Mark gaps > 500ms
                ax.axvspan(timestamps[j], timestamps[j + 1], alpha=0.1,
                           color='red', label='Time gap' if j == 0 else "")
                # Add text annotation
                mid_time = (timestamps[j] + timestamps[j + 1]) / 2
                ax.text(mid_time, ax.get_ylim()[1] * 0.95, f'{gap:.1f}s gap',
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_ylabel('Range (m)', fontsize=11)
        ax.set_title(f'Track {track_id}: Continuous Tracking Through Time Gaps', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set x-axis to show actual time
        ax.set_xlim(min(dense_times) - 0.5, max(dense_times) + 0.5)

    if len(axes) > 0:
        axes[-1].set_xlabel('Time (seconds)', fontsize=11)

    plt.suptitle('Temporal Evolution: Real-Time Tracking with Predictions\n'
                 'Tracker maintains position estimates during measurement gaps',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = Path(output_dir) / "tracking_temporal_evolution.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_timing_analysis(frame_times: List[Tuple[int, float]],
                              time_gaps: List[float],
                              output_dir: str):
    """Visualize frame timing and gaps."""
    prepare_output_directories(output_dir)

    frames, timestamps = zip(*frame_times)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Timestamps vs Frame ID
    ax1.plot(frames, timestamps, 'b-', linewidth=1)
    ax1.set_xlabel('Frame ID')
    ax1.set_ylabel('Timestamp (seconds)')
    ax1.set_title('Frame Timestamps Showing Temporal Gaps')
    ax1.grid(True, alpha=0.3)

    # Highlight large gaps
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > 0.5:  # Gaps > 0.5s
            ax1.axvspan(frames[i - 1], frames[i], alpha=0.2, color='red')

    # 2. Time gaps histogram
    ax2.hist(time_gaps, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Time Gap (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Time Gaps Between Frames')
    ax2.axvline(np.mean(time_gaps), color='red', linestyle='--',
                label=f'Mean: {np.mean(time_gaps):.3f}s')
    ax2.legend()

    # 3. Time gaps over frame sequence
    ax3.plot(frames[1:], time_gaps, 'g-', linewidth=1)
    ax3.set_xlabel('Frame ID')
    ax3.set_ylabel('Time Gap to Previous Frame (s)')
    ax3.set_title('Time Gaps Throughout Sequence')
    ax3.grid(True, alpha=0.3)

    # Mark large gaps
    large_gap_threshold = np.percentile(time_gaps, 95)
    large_gaps = [(frames[i + 1], gap) for i, gap in enumerate(time_gaps)
                  if gap > large_gap_threshold]
    if large_gaps:
        gap_frames, gap_values = zip(*large_gaps)
        ax3.scatter(gap_frames, gap_values, color='red', s=50, zorder=5,
                    label=f'Large gaps (>{large_gap_threshold:.2f}s)')
        ax3.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'timing_analysis.png', dpi=150)
    plt.close()


def visualize_tracking_during_gaps(all_tracks: List[List[Track]],
                                   frame_times: List[Tuple[int, float]],
                                   gap_threshold: float,
                                   output_dir: str):
    """Visualize how tracking maintains estimates during time gaps."""
    prepare_output_directories(output_dir)

    # Find frames with large preceding gaps
    gap_frames = []
    for i in range(1, len(frame_times)):
        time_gap = frame_times[i][1] - frame_times[i - 1][1]
        if time_gap > gap_threshold:
            gap_frames.append({
                'before_idx': i - 1,  # Store index, not frame_id
                'after_idx': i,
                'before': frame_times[i - 1],
                'after': frame_times[i],
                'gap': time_gap
            })

    if not gap_frames:
        print(f"No gaps larger than {gap_threshold}s found")
        return

    # Visualize tracking through largest gap
    largest_gap = max(gap_frames, key=lambda x: x['gap'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get indices and frame info
    before_idx = largest_gap['before_idx']
    after_idx = largest_gap['after_idx']
    before_frame_id = largest_gap['before'][0]
    after_frame_id = largest_gap['after'][0]
    gap_duration = largest_gap['gap']

    # Get tracks from the correct indices
    before_tracks = all_tracks[before_idx] if before_idx < len(all_tracks) else []
    after_tracks = all_tracks[after_idx] if after_idx < len(all_tracks) else []

    # Left plot: Position tracking through gap
    ax1 = axes[0]

    # Track if we've added any data to the plot
    has_data = False

    # Plot track positions and predicted trajectories
    for track_idx, track in enumerate(before_tracks):
        if track is None or not hasattr(track, 'state'):
            continue

        x, y = track.position
        vx, vy = track.velocity

        # Plot current position
        ax1.scatter(x, y, color='blue', s=100, marker='o',
                    label='Position before gap' if track_idx == 0 else "")
        has_data = True

        # Plot predicted trajectory during gap
        t_points = np.linspace(0, gap_duration, 20)
        x_pred = x + vx * t_points
        y_pred = y + vy * t_points
        ax1.plot(x_pred, y_pred, 'b--', alpha=0.5, linewidth=2,
                 label='Kalman prediction' if track_idx == 0 else "")

        # Find matching track after gap
        matched = False
        for after_track in after_tracks:
            if after_track is not None and after_track.id == track.id:
                x_after, y_after = after_track.position
                ax1.scatter(x_after, y_after, color='green', s=100, marker='s',
                            label='Position after gap' if track_idx == 0 else "")
                ax1.plot([x_pred[-1], x_after], [y_pred[-1], y_after],
                         'r-', linewidth=2, label='Correction' if track_idx == 0 else "")
                matched = True
                break

        # If no match found, show where prediction ended
        if not matched:
            ax1.scatter(x_pred[-1], y_pred[-1], color='orange', s=100, marker='x',
                        label='Lost track' if track_idx == 0 else "")

    # If no tracks found, add informative text
    if not has_data:
        ax1.text(0.5, 0.5, f'No active tracks found\nat frame {before_frame_id}',
                 transform=ax1.transAxes, ha='center', va='center',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Track Maintenance During {gap_duration:.2f}s Gap\n'
                  f'Frames {before_frame_id} → {after_frame_id}')
    if has_data:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Add some margin if we have data
    if has_data:
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        x_margin = (xlim[1] - xlim[0]) * 0.1
        y_margin = (ylim[1] - ylim[0]) * 0.1
        ax1.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
        ax1.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)

    # Right plot: Uncertainty growth
    ax2 = axes[1]

    # Get actual process noise from a track's covariance if available
    if before_tracks and hasattr(before_tracks[0], 'covariance'):
        # Extract position variance from covariance matrix
        initial_variance = (before_tracks[0].covariance[0, 0] +
                            before_tracks[0].covariance[1, 1]) / 2
    else:
        initial_variance = 2.0  # Default

    # Simulate uncertainty growth during gap
    time_points = np.linspace(0, gap_duration, 50)

    # Use actual process noise from Kalman filter if available
    process_noise = 10.0  # Default value

    # More realistic uncertainty growth model
    # Uncertainty grows with sqrt of time for random walk
    variance_growth = initial_variance + process_noise * time_points

    ax2.plot(time_points, np.sqrt(variance_growth), 'r-', linewidth=2,
             label='Position uncertainty (1σ)')
    ax2.fill_between(time_points, 0, np.sqrt(variance_growth),
                     alpha=0.2, color='red')

    # Add 2-sigma and 3-sigma bounds
    ax2.plot(time_points, 2 * np.sqrt(variance_growth), 'r--',
             alpha=0.5, linewidth=1, label='2σ bound')
    ax2.plot(time_points, 3 * np.sqrt(variance_growth), 'r:',
             alpha=0.5, linewidth=1, label='3σ bound')

    ax2.set_xlabel('Time into gap (s)')
    ax2.set_ylabel('Position Uncertainty (m)')
    ax2.set_title('Kalman Filter Uncertainty Growth During Gap')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tracking_through_gaps.png', dpi=150)
    plt.close()

    # Also create a summary of all gaps
    if len(gap_frames) > 1:
        fig2, ax = plt.subplots(figsize=(10, 6))

        gaps = [g['gap'] for g in gap_frames]
        frame_ids = [g['before'][0] for g in gap_frames]

        ax.bar(range(len(gaps)), gaps, color='blue', alpha=0.7)
        ax.set_xlabel('Gap Index')
        ax.set_ylabel('Gap Duration (s)')
        ax.set_title(f'All Time Gaps > {gap_threshold}s in Sequence')

        # Add frame IDs as labels
        for i, (gap, fid) in enumerate(zip(gaps, frame_ids)):
            ax.text(i, gap + 0.01, f'Frame {fid}', ha='center', va='bottom', fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'all_time_gaps.png', dpi=150)
        plt.close()