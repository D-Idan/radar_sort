# offline_tracking.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from radar_tracking import TrackletManager, Detection

def setup_tracking_system():
    """
    Initialize and return a TrackletManager and its config dict.
    """
    tracker_config = {
        'max_age': 5,        # frames to keep a track alive without new detections
        'min_hits': 3,       # how many hits before we “confirm” a track
        'iou_threshold': 8.0,  # maximum distance (meters) for associating dets→tracks
        'dt': 0.1            # assumed time step between frames (you can adjust)
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
    We’ll use numSample as sample_id, and radar_R_m, radar_A_deg as ground‐truth.
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
            confidence=1.0,           # ground truth = perfect confidence
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
    Main offline tracking function.
    - Reads predictions + labels
    - Slides through each unique frame_id in ascending order
    - Updates TrackletManager per frame
    - Records per‐frame tracks into a list, then writes 'tracking.csv'
    - Generates simple visualizations at the end
    """
    # 1) Load data
    preds_df = load_predictions(preds_csv)
    labels_df = load_labels(labels_csv)

    # 2) Determine all unique frame IDs (sample_id) in ascending order
    all_frames = sorted(preds_df['sample_id'].unique().tolist())

    # 3) Initialize tracker
    tracker, config = setup_tracking_system()

    # 4) Prepare to collect tracking output
    # We'll keep a list of dicts, each representing one "track‐detection" assignment row
    tracking_rows = []

    # Also for visualization: we want #detections_per_frame and #tracks_per_frame
    det_counts = []
    track_counts = []

    # 5) Loop over frames
    for fid in all_frames:
        # a) Build Detection list from predictions
        detections = build_detections_for_frame(preds_df, fid)
        det_counts.append(len(detections))

        # b) Build ground truth list (optional—but we include to mirror original)
        ground_truth = build_ground_truth_for_frame(labels_df, fid)

        # c) Update tracker
        active_tracks = tracker.update(detections, ground_truth)

        # d) Record how many confirmed tracks are active this frame
        track_counts.append(len(active_tracks))

        # e) For each track, extract needed fields to save in CSV
        # We must output: sample_id, frame_id, track_id, detection_id, confidence, range_m, azimuth_deg,
        # track_age, hits, track_state, x1, y1, x2, y2, x3, y3, x4, y4
        for det, track in zip(detections, tracker._last_associations[detection_index_of(detections):]):
            # The above line assumes TrackletManager stores associations in the same order as `detections`.
            # If your TrackletManager API differs, adjust accordingly to find which detection→track mapping.
            pass

        # A safer approach: directly iterate through tracker.tracklets (if TrackletManager exposes them),
        # and for each active Tracklet object pull its most recent Detection.
        #
        # Below is a generic template—you may need to adapt field‐names to your Tracklet class.
        for trk in tracker.tracklets:
            # Only record if the track is currently "confirmed" or "tentative"; skip if “deleted”
            if trk.state == TrackletManager.TrackState.TENTATIVE or trk.state == TrackletManager.TrackState.CONFIRMED:
                latest_det: Detection = trk.detections[-1]
                row = {
                    'sample_id': int(fid),
                    'frame_id': int(fid),
                    'track_id': int(trk.id),
                    'detection_id': int(getattr(latest_det, '_detection_id', -1)),
                    'confidence': float(latest_det.confidence),
                    'range_m': float(latest_det.range_m),
                    'azimuth_deg': float(np.degrees(latest_det.azimuth_rad)),
                    'track_age': int(trk.age),
                    'hits': int(trk.hits),
                    'track_state': trk.state.name,  # e.g., 'CONFIRMED' or 'TENTATIVE'
                    # If your Tracklet stores bounding boxes, replace x1..y4 fields accordingly.
                    # Here we simply duplicate range/azimuth into x1..y4 as placeholders:
                    'x1': np.nan, 'y1': np.nan,
                    'x2': np.nan, 'y2': np.nan,
                    'x3': np.nan, 'y3': np.nan,
                    'x4': np.nan, 'y4': np.nan
                }
                tracking_rows.append(row)

    # 6) Once all frames are done, save tracking output
    track_df = pd.DataFrame(tracking_rows)
    # Ensure column order:
    cols = [
        'sample_id', 'frame_id', 'track_id', 'detection_id', 'confidence',
        'range_m', 'azimuth_deg', 'track_age', 'hits', 'track_state',
        'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
    ]
    track_df = track_df[cols]
    track_df.to_csv(output_tracking_csv, index=False)

    # 7) Build some simple visualizations
    #    a) # detections vs. # tracks per frame
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(all_frames, det_counts, label='Detections per Frame', marker='o', linestyle='-')
    ax.plot(all_frames, track_counts, label='Confirmed Tracks per Frame', marker='s', linestyle='--')
    ax.set_xlabel("Frame (sample_id)")
    ax.set_ylabel("Count")
    ax.set_title("Detections vs. Confirmed Tracks Over Frames")
    ax.legend()
    plt.tight_layout()
    fig.savefig("counts_per_frame.png")
    plt.close(fig)

    #    b) Histogram of track lengths (from TrackletManager summary, if available)
    summary = tracker.get_tracking_summary()
    # Assume summary contains a list of all final track lengths under key 'track_lengths'
    if 'track_lengths' in summary:
        lengths = np.array(summary['track_lengths'])
    else:
        # Fallback: iterate through tracklets and grab their hit count or age
        lengths = np.array([trk.hits for trk in tracker.tracklets])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(lengths, bins=20, edgecolor='black')
    ax2.set_xlabel("Track Length (frames)")
    ax2.set_ylabel("Number of Tracks")
    ax2.set_title("Histogram of Track Lengths")
    plt.tight_layout()
    fig2.savefig("track_length_histogram.png")
    plt.close(fig2)

    #    c) Example trajectories: take first few confirmed track IDs and plot range vs. azimuth over time
    # Build a dict: track_id → list of (timestamp, range, azimuth)
    trajs: dict[int, list[tuple[float, float, float]]] = {}
    for fid in all_frames:
        detections = build_detections_for_frame(preds_df, fid)
        # We have to re‐run the tracker to get the detection→track associations
        # (A more efficient approach would have recorded these associations in step 5,e, but
        # for simplicity we re-use tracker.tracklets themselves.)
        # So instead, just loop through all tracklets and find which ones have a detection at this timestamp.
        for trk in tracker.tracklets:
            for det in trk.detections:
                if abs(det.timestamp - fid) < 1e-6:
                    tid = trk.id
                    if tid not in trajs:
                        trajs[tid] = []
                    trajs[tid].append((fid, det.range_m, np.degrees(det.azimuth_rad)))
    # Pick up to 3 tracks to plot
    example_ids = list(trajs.keys())[:3]
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for tid in example_ids:
        data = sorted(trajs[tid], key=lambda x: x[0])
        times = [x[0] for x in data]
        ranges = [x[1] for x in data]
        azs = [x[2] for x in data]
        ax3.plot(times, ranges, marker='o', label=f'Track {tid} (range)')
        ax3.set_xlabel("Frame (sample_id)")
        ax3.set_ylabel("Range (m)")
        ax3_twin = ax3.twinx()
        ax3_twin.plot(times, azs, marker='x', linestyle='--', label=f'Track {tid} (azimuth)')
        ax3_twin.set_ylabel("Azimuth (°)")
    ax3.set_title("Example Track Trajectories (Range & Azimuth over Time)")
    lines, labels = [], []
    for ax in (ax3, ax3_twin):
        line, lab = ax.get_legend_handles_labels()
        lines += line
        labels += lab
    ax3.legend(lines, labels, loc='upper right', fontsize='small')
    plt.tight_layout()
    fig3.savefig("example_trajectories.png")
    plt.close(fig3)

    print(f"Offline tracking done. Saved:\n"
          f"  - {output_tracking_csv}\n"
          f"  - counts_per_frame.png\n"
          f"  - track_length_histogram.png\n"
          f"  - example_trajectories.png")

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
