# offline_tracking.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path

from tqdm import tqdm

from radar_tracking import TrackletManager, Detection, Track
from radar_tracking.track_viz import (
    visualize_counts_vs_tracks_per_frame,
    prepare_viz_directory,
    visualize_frame_radar_azimuth,
    visualize_tracklet_lifetime_histogram,
    visualize_avg_confidence_over_time,
    visualize_all_frames_overview,
    visualize_tracking_temporal_evolution
)


def setup_tracking_system():
    """
    Initialize and return a TrackletManager and its config dict.
    """
    tracker_config = {
        'max_age': 5,  # frames to keep a track alive without new detections
        'min_hits': 3,  # how many hits before we "confirm" a track
        'iou_threshold': 0.1,  # maximum distance (meters) for associating dets→tracks
        'dt': 0.1  # assumed time step between frames (you can adjust)
    }
    manager = TrackletManager(tracker_config=tracker_config)
    return manager, tracker_config


def load_predictions(pred_csv_path: str) -> pd.DataFrame:
    """
    Load all_predictions.csv into a DataFrame and sort by sample_id/frame order.
    Expects columns: detection_id, confidence, x1, y1, …, range_m, azimuth_deg, sample_id
    """
    df = pd.read_csv(pred_csv_path, sep=r'\s+|,', engine='python')
    # Ensure it's sorted by sample_id (i.e., the frame order)
    df = df.sort_values(by='sample_id').reset_index(drop=True)
    return df


def load_labels(label_csv_path: str) -> pd.DataFrame:
    """
    Load labels.csv into a DataFrame. Expects columns including: numSample, radar_R_m, radar_A_deg, etc.
    We'll use numSample as sample_id, and radar_R_m, radar_A_deg as ground‐truth.
    """
    df = pd.read_csv(label_csv_path, sep=r'\s+|,', engine='python')
    return df


def build_detections_for_frame(
        preds_df: pd.DataFrame,
        frame_id: int
) -> list[Detection]:
    """
    Given the full predictions DataFrame and a specific frame_id (sample_id),
    return a list of Detection objects for that frame.
    """
    # Filter all rows whose sample_id == frame_id
    sub = preds_df[preds_df['sample_id'] == frame_id]
    dets: list[Detection] = []
    for _, row in sub.iterrows():
        r = float(row['range_m'])
        az_deg = float(row['azimuth_deg'])
        det = Detection(
            range_m=r,
            azimuth_rad=np.radians(az_deg),
            confidence=float(row['confidence']),
            timestamp=float(frame_id)
        )
        # We attach detection_id in a custom attribute so we can save later
        det._detection_id = int(row['detection_id'])
        dets.append(det)
    return dets


def build_ground_truth_for_frame(
        labels_df: pd.DataFrame,
        frame_id: int
) -> list[Detection]:
    """
    Convert ground‐truth (labels.csv) for that frame_id into Detection objects.
    Assumes columns: numSample (→ sample_id), radar_R_m, radar_A_deg.
    """
    sub = labels_df[labels_df['numSample'] == frame_id]
    gt_list: list[Detection] = []
    for _, row in sub.iterrows():
        r = float(row['radar_R_m'])
        az_deg = float(row['radar_A_deg'])
        gt = Detection(
            range_m=r,
            azimuth_rad=np.radians(az_deg),
            confidence=1.0,  # ground truth = perfect confidence
            timestamp=float(frame_id)
        )
        gt_list.append(gt)
    return gt_list


def offline_tracking(
        preds_csv: str,
        labels_csv: str,
        output_tracking_csv: str
):
    """
    Main offline‐tracking function:
      1) Reads predictions + labels
      2) Iterates over each unique frame_id in ascending order
      3) Builds Detection objects, calls tracker.update(...)
      4) Writes out one row per active track each frame into tracking.csv
      5) Saves comprehensive visualizations (PNG files)
    """
    # 1) Load all_predictions.csv and labels.csv
    preds_df = load_predictions(preds_csv)
    labels_df = load_labels(labels_csv)

    # 2) Unique frame IDs (sample_id) in sorted order
    all_frames = sorted(preds_df['sample_id'].unique().tolist())

    # 3) Initialize tracker
    tracker, config = setup_tracking_system()

    # For storing the CSV rows
    tracking_rows = []

    # For visualization data storage
    det_counts: List[int] = []
    track_counts: List[int] = []
    avg_confidence_per_frame: List[float] = []

    # Store all data for comprehensive visualizations
    all_detections: List[List[Detection]] = []
    all_ground_truth: List[List[Detection]] = []
    all_tracks: List[List[Track]] = []

    # Visualization setup
    viz_dir = "visualizations_radar"
    prepare_viz_directory(viz_dir)

    # 4) Loop over frames
    for frame_id in tqdm(all_frames,
                         total=len(all_frames),
                         desc="Processing Frames",
                         unit="frame",
                         colour="green",
                         dynamic_ncols=True):
        # a) Build detections for this frame
        detections = build_detections_for_frame(preds_df, frame_id)
        det_counts.append(len(detections))

        # b) Build ground truth for this frame
        ground_truth = build_ground_truth_for_frame(labels_df, frame_id)

        # c) Update tracker
        active_tracks = tracker.update(detections, ground_truth)

        # Store data for comprehensive visualizations
        all_detections.append(detections)
        all_ground_truth.append(ground_truth)
        all_tracks.append(active_tracks.copy())  # Make a copy to preserve state

        # Count how many tracks are currently "confirmed" or "tentative"
        track_counts.append(len(active_tracks))

        # Compute average confidence among active tracks this frame
        if active_tracks:
            confidences = [t.last_detection.confidence for t in active_tracks if t.last_detection is not None]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
        else:
            avg_conf = 0.0
        avg_confidence_per_frame.append(avg_conf)

        # d) Record one row per active track
        for track in active_tracks:
            # Last detection from this track
            last_det = track.last_detection
            if last_det is None:
                continue

            rid = getattr(last_det, '_detection_id', -1)
            r = float(last_det.range_m)
            az = float(np.degrees(last_det.azimuth_rad))
            conf = float(last_det.confidence)

            row = {
                'sample_id': int(frame_id),
                'frame_id': int(frame_id),
                'track_id': int(track.id),
                'detection_id': int(rid),
                'confidence': conf,
                'range_m': r,
                'azimuth_deg': az,
                'track_age': int(track.age),
                'hits': int(track.hits),
                'track_state': track.state.name if hasattr(track.state, 'name') else str(track.state),
                # x1..y4 left as NaN placeholders; replace if you have pixel‐corner data
                'x1': np.nan, 'y1': np.nan,
                'x2': np.nan, 'y2': np.nan,
                'x3': np.nan, 'y3': np.nan,
                'x4': np.nan, 'y4': np.nan
            }
            tracking_rows.append(row)

        # e) Visualize this frame (individual frame visualization)
        visualize_frame_radar_azimuth(
            frame_id=frame_id,
            detections=detections,
            ground_truth=ground_truth,
            active_tracks=active_tracks,
            viz_dir=viz_dir
        )

    # 5) Build DataFrame and write tracking.csv
    track_df = pd.DataFrame(tracking_rows)
    cols = [
        'sample_id', 'frame_id', 'track_id', 'detection_id', 'confidence',
        'range_m', 'azimuth_deg', 'track_age', 'hits', 'track_state',
        'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
    ]
    track_df = track_df[cols]
    track_df.to_csv(output_tracking_csv, index=False)

    # 6) Generate all visualizations using the visualization functions

    # a) Network Output per Frame vs. Confirmed Tracks per Frame
    visualize_counts_vs_tracks_per_frame(
        all_frames=all_frames,
        det_counts=det_counts,
        track_counts=track_counts,
        path_save="counts_vs_tracks_per_frame.png"
    )

    # b) Histogram of Tracklet Lifetimes
    visualize_tracklet_lifetime_histogram(
        tracker=tracker,
        path_save="tracklet_lifetime_histogram.png"
    )

    # c) Average Confidence of Active Tracks Over Time
    visualize_avg_confidence_over_time(
        all_frames=all_frames,
        avg_confidence_per_frame=avg_confidence_per_frame,
        path_save="avg_confidence_over_time.png"
    )

    # d) NEW: Comprehensive overview showing all frames data
    visualize_all_frames_overview(
        all_detections=all_detections,
        all_ground_truth=all_ground_truth,
        all_tracks=all_tracks,
        all_frames=all_frames,
        path_save="all_frames_overview.png"
    )

    # e) NEW: Temporal evolution visualization
    visualize_tracking_temporal_evolution(
        all_detections=all_detections,
        all_ground_truth=all_ground_truth,
        all_tracks=all_tracks,
        all_frames=all_frames,
        path_save="tracking_temporal_evolution.png"
    )

    print(f"\nOffline tracking completed. Files written:\n"
          f"  • {output_tracking_csv}\n"
          f"  • counts_vs_tracks_per_frame.png\n"
          f"  • tracklet_lifetime_histogram.png\n"
          f"  • avg_confidence_over_time.png\n"
          f"  • all_frames_overview.png\n"
          f"  • tracking_temporal_evolution.png\n"
          f"  • Individual frame visualizations in {viz_dir}/\n")

if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(
    #     description="Run offline tracking on all_predictions.csv and generate simple visualizations."
    # )
    # parser.add_argument(
    #     "--preds", type=str, required=True,
    #     help="Path to all_predictions.csv"
    # )
    # parser.add_argument(
    #     "--labels", type=str, required=True,
    #     help="Path to labels.csv (ground truth)"
    # )
    # parser.add_argument(
    #     "--out", type=str, default="tracking.csv",
    #     help="Output CSV filename for tracking results"
    # )
    # args = parser.parse_args()

    from pathlib import Path
    import json

    path_repo = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils')
    if not path_repo.exists():
        path_repo = Path('/mnt/data/myprojects/PycharmProjects/thesis_repos/radar_sort/utils')

    path_config_default = path_repo / Path('T_FFTRadNet/RadIal/ADCProcessing/data_config.json')
    config = json.load(open(path_config_default))
    record = config['target_value']
    root_folder = Path(config['Data_Dir'], 'RadIal_Data',record)
    labels_csv = Path(root_folder, 'labels.csv')

    path_file_par = Path(__file__).parent

    args = {
        'preds_csv': path_file_par / Path('./predictions/all_predictions.csv'),
        'labels_csv': Path(labels_csv),
        'output_tracking_csv': path_file_par / Path('./predictions/tracking.csv'),
    }

    offline_tracking(**args)
