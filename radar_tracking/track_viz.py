import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from radar_tracking import Detection, Track

def prepare_viz_directory(viz_dir: str):
    """
    Create viz_dir if it doesn’t exist.
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
      • detections: scatter as blue dots
      • ground_truth: scatter as green X’s
      • active_tracks: red circles with track IDs labeled

    Saves to: {viz_dir}/frame_{frame_id:06d}.jpg
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # 1) Plot predicted detections
    if detections:
        az_det = [np.degrees(d.azimuth_rad) for d in detections]
        rng_det = [d.range_m for d in detections]
        ax.scatter(az_det, rng_det, c='blue', s=20, label='Predicted Detections', alpha=0.6)

    # 2) Plot ground truth
    if ground_truth:
        az_gt = [np.degrees(d.azimuth_rad) for d in ground_truth]
        rng_gt = [d.range_m for d in ground_truth]
        ax.scatter(az_gt, rng_gt, c='green', marker='x', s=40, label='Ground Truth')

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
    ax.set_xlim(-45, 45)   # adjust as needed for your sensor’s FoV
    ax.set_ylim(0, 120)    # adjust to your max radar range

    plt.tight_layout()
    out_path = os.path.join(viz_dir, f"frame_{frame_id:06d}.jpg")
    plt.savefig(out_path, dpi=150)
    plt.close()


# Detections per Frame vs. Confirmed Tracks per Frame
def visualize_counts_vs_tracks_per_frame(all_frames: List[int],
                                         det_counts: List[int],
                                         track_counts: List[int],
                                         path_save: str = "counts_vs_tracks_per_frame.png",
                                         ):
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(all_frames, det_counts, marker='o', linestyle='-', label='Detections/frame')
    ax1.plot(all_frames, track_counts, marker='s', linestyle='--', label='Active tracks/frame')
    ax1.set_xlabel("Frame ID (sample_id)")
    ax1.set_ylabel("Count")
    ax1.set_title("Detections vs. Active Tracks per Frame")
    ax1.legend(loc='upper right')
    plt.tight_layout()
    fig1.savefig(path_save)
    plt.close(fig1)