# main_processing_with_tracking.py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
import json
import pickle
from typing import Dict, List, Any

# Import your existing modules
from utils.enhanced_save_model_outputs import (
    save_comprehensive_results,
    analyze_tracking_benefits,
    convert_model_output_to_detections,
    batch_save_predictions
)
from radar_tracking import TrackletManager, Detection
from plots import dl_output_viz


def extract_ground_truth_detections(data, frame_timestamp: float) -> List[Detection]:
    """
    Extract ground truth detections from your data structure.

    Args:
        data: Your data tuple [radar_FFT, segmap, out_label, box_labels, image]
        frame_timestamp: Timestamp for this frame

    Returns:
        List of ground truth Detection objects
    """
    ground_truth = []

    # Extract from your data structure - data[3] appears to be box_labels
    box_labels = data[3] if len(data) > 3 else None

    if box_labels is not None:
        try:
            # Convert your box labels to Detection objects
            # You may need to modify this based on your actual label format
            if hasattr(box_labels, '__iter__') and not isinstance(box_labels, str):
                for label in box_labels:
                    try:
                        # Assuming box_labels contains range/azimuth information
                        # Modify this based on your actual label format
                        if hasattr(label, 'get'):
                            range_m = float(label.get('range', 0))
                            azimuth_deg = float(label.get('azimuth', 0))
                        elif isinstance(label, (list, tuple)) and len(label) >= 2:
                            range_m = float(label[0])
                            azimuth_deg = float(label[1])
                        else:
                            continue

                        azimuth_rad = np.radians(azimuth_deg)

                        gt_detection = Detection(
                            range_m=range_m,
                            azimuth_rad=azimuth_rad,
                            confidence=1.0,  # Ground truth has perfect confidence
                            timestamp=frame_timestamp
                        )
                        ground_truth.append(gt_detection)
                    except (ValueError, TypeError, KeyError, IndexError):
                        continue
        except Exception:
            pass  # Handle cases where ground truth is not available or malformed

    return ground_truth


def setup_tracking_system():
    """Initialize the tracking system with configuration"""
    tracker_config = {
        'max_age': 5,  # Maximum frames to keep track alive without detections
        'min_hits': 3,  # Minimum detections before track is considered confirmed
        'iou_threshold': 8.0,  # Maximum distance for association (meters)
        'dt': 0.1  # Time step between frames (adjust based on your frame rate)
    }

    tracklet_manager = TrackletManager(tracker_config=tracker_config)
    return tracklet_manager, tracker_config


def initialize_results_storage(tracker_config):
    """Initialize storage for results comparison"""
    return {
        'without_tracking': {
            'model_outputs': {},
            'detections_per_frame': {},
            'total_detections': 0,
            'processing_times': []
        },
        'with_tracking': {
            'tracks_per_frame': {},
            'tracklet_statistics': {},
            'total_tracks_created': 0,
            'processing_times': [],
            'tracking_metrics': []
        },
        'metadata': {
            'tracker_config': tracker_config,
            'total_frames': 0,
            'dataset_info': {}
        }
    }


def process_visualization(data, outputs, enc):
    """Handle the visualization processing (your existing code)"""
    if data[4] is not None:  # there is image
        try:
            path_repo = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils')
            dd = "/Volumes/ELEMENTS/datasets/radial"
            record = "RECORD@2020-11-22_12.45.05"

            if not path_repo.exists():
                path_repo = Path('/mnt/data/myprojects/PycharmProjects/thesis_repos/radar_sort/utils')
                dd = "/mnt/data/datasets/radial/gd/raw_data/"
                record = "RECORD@2020-11-22_12.37.16"

            root_folder = Path(dd, 'RadIal_Data', record)
            ra_dir = Path(root_folder, 'radar_RA')
            ra_path = Path(ra_dir) / f"ra_{data[-1]:06d}.npy"

            if ra_path.exists():
                ra_map = np.load(ra_path)
                res_ra = dl_output_viz.visualize_detections_on_bev(ra_map, outputs, enc, max_range=103)
                dl_output_viz.draw_boxes_on_RA_map(res_ra)

        except Exception as e:
            print(f"Visualization warning for frame {data[5] if len(data) > 5 else 'unknown'}: {e}")


def main_processing_with_tracking(net, dataset, config, checkpoint_filename, enc, device, viz_jit=False):
    """
    Main processing function that runs inference with and without tracking

    Args:
        net: Your neural network model
        dataset: Your dataset
        config: Configuration dictionary
        checkpoint_filename: Path to model checkpoint
        enc: Encoder for decoding predictions
        device: Device to run on (cuda/cpu)
    """

    print("=" * 80)
    print("RADAR DETECTION AND TRACKING PROCESSING")
    print("=" * 80)

    # Initialize model (your existing code)
    net.to(device)
    dict_checkpoint = torch.load(checkpoint_filename, weights_only=False, map_location=torch.device(device))
    net.load_state_dict(dict_checkpoint['net_state_dict'])
    net = net.double()
    net.eval()

    print(f"Model loaded from: {checkpoint_filename}")
    print(f"Running on device: {device}")
    print(f"Data mode: {config.get('data_mode', 'unknown')}")

    # Initialize tracking system
    tracklet_manager, tracker_config = setup_tracking_system()
    print(f"Tracking system initialized with config: {tracker_config}")

    # Initialize results storage
    results_comparison = initialize_results_storage(tracker_config)

    # Storage for original batch saving (your existing system)
    model_outputs_dict = {}

    # Main processing loop with better error handling
    frame_count = 0
    successful_frames = 0
    failed_frames = 0

    print(f"\nStarting processing...")

    for data in tqdm(dataset, desc="Processing samples", unit="sample"):
        try:
            # === ORIGINAL MODEL INFERENCE ===
            inference_start_time = time.time()

            inputs = torch.tensor(data[0]).permute(2, 0, 1).to(device).unsqueeze(0)

            with torch.set_grad_enabled(False):
                outputs = net(inputs)
                if config['data_mode'] == 'ADC':
                    intermediate = net.DFT(inputs).detach().cpu().numpy()[0]
                else:
                    intermediate = None

            inference_time = time.time() - inference_start_time

            # Store original results
            sample_id = data[5] if len(data) > 5 else frame_count
            model_outputs_dict[sample_id] = outputs
            results_comparison['without_tracking']['processing_times'].append(inference_time)

            # === CONVERT TO DETECTIONS FOR TRACKING ===
            frame_timestamp = float(frame_count)
            detections = convert_model_output_to_detections(outputs, enc, frame_timestamp)
            ground_truth = extract_ground_truth_detections(data, frame_timestamp)

            # Store detections without tracking
            results_comparison['without_tracking']['detections_per_frame'][frame_count] = {
                'detections': detections,
                'ground_truth': ground_truth,
                'num_detections': len(detections),
                'frame_id': sample_id
            }
            results_comparison['without_tracking']['total_detections'] += len(detections)

            # === APPLY TRACKING ===
            tracking_start_time = time.time()
            tracks = tracklet_manager.update(detections, ground_truth)
            tracking_time = time.time() - tracking_start_time

            results_comparison['with_tracking']['processing_times'].append(tracking_time)
            results_comparison['with_tracking']['tracks_per_frame'][frame_count] = {
                'tracks': tracks,
                'num_tracks': len(tracks),
                'frame_id': sample_id,
                'detections_input': detections,
                'ground_truth': ground_truth
            }

            # === VISUALIZATION ===
            if viz_jit:
                try:
                    process_visualization(data, outputs, enc)
                except Exception as viz_e:
                    print(f"Visualization warning for frame {frame_count}: {viz_e}")

            successful_frames += 1

        except Exception as e:
            import traceback
            failed_frames += 1
            print(f"\nERROR processing frame {frame_count}:")
            print(f"Sample ID: {data[5] if len(data) > 5 else 'unknown'}")
            print(f"Data shape: {[np.array(d).shape if hasattr(d, 'shape') else type(d) for d in data]}")
            print(f"Exception: {e}")
            print("Full traceback:")
            traceback.print_exc()
            print("-" * 80)

            # Continue processing other frames
            continue

        finally:
            frame_count += 1

            # Progress update every 50 frames
            if frame_count % 50 == 0:
                print(
                    f"\nProgress: {frame_count} frames processed, {successful_frames} successful, {failed_frames} failed")

    print(f"\nProcessing summary: {successful_frames}/{frame_count} frames successful")

    # === FINALIZE RESULTS ===
    print(f"\nProcessing completed. Total frames processed: {frame_count}")

    # Get final tracking statistics
    tracking_summary = tracklet_manager.get_tracking_summary()
    results_comparison['with_tracking']['tracklet_statistics'] = tracking_summary
    results_comparison['with_tracking']['total_tracks_created'] = tracking_summary.get('total_tracklets', 0)

    # Store model outputs for enhanced results
    results_comparison['without_tracking']['model_outputs'] = model_outputs_dict

    # Store metadata
    results_comparison['metadata']['total_frames'] = frame_count
    results_comparison['metadata']['dataset_info'] = {
        'config_data_mode': config.get('data_mode', 'unknown'),
        'checkpoint_file': checkpoint_filename,
        'total_samples': len(dataset) if hasattr(dataset, '__len__') else frame_count
    }

    # === PRINT COMPARISON SUMMARY ===
    print_processing_summary(results_comparison, frame_count)

    # === SAVE RESULTS ===
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # 1. Save original batch predictions (your existing system)
    print("Saving original predictions (existing format)...")
    batch_save_predictions(model_outputs_dict, enc, "plots/predictions/")

    # 2. Save comprehensive results using enhanced system
    print("Saving comprehensive tracking results...")
    saved_dataframes = save_comprehensive_results(results_comparison, enc, "plots/tracking_results/")

    # 3. Analyze and report tracking benefits
    print("Analyzing tracking benefits...")
    tracking_analysis = analyze_tracking_benefits(results_comparison, "plots/tracking_results/")

    # 4. Print final tracklet manager summary
    print("\n" + "=" * 60)
    print("FINAL TRACKING SUMMARY")
    print("=" * 60)
    tracklet_manager.print_summary()

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results saved in:")
    print(f"  - plots/predictions/ (original format)")
    print(f"  - plots/tracking_results/ (comprehensive analysis)")

    return {
        'results_comparison': results_comparison,
        'tracking_analysis': tracking_analysis,
        'saved_dataframes': saved_dataframes,
        'tracklet_manager': tracklet_manager
    }


def print_processing_summary(results_comparison, frame_count):
    """Print a comprehensive summary of processing results"""
    print("\n" + "=" * 80)
    print("RADAR DETECTION vs TRACKING COMPARISON")
    print("=" * 80)

    print(f"Total Frames Processed: {frame_count}")

    # Without tracking stats
    total_detections = results_comparison['without_tracking']['total_detections']
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    avg_inference_time = np.mean(results_comparison['without_tracking']['processing_times']) if \
    results_comparison['without_tracking']['processing_times'] else 0

    print(f"\nWITHOUT TRACKING:")
    print(f"  Total Detections: {total_detections}")
    print(f"  Avg Detections per Frame: {avg_detections:.2f}")
    print(f"  Avg Processing Time: {avg_inference_time:.4f}s ({avg_inference_time * 1000:.2f}ms)")

    # With tracking stats
    tracking_summary = results_comparison['with_tracking']['tracklet_statistics']
    total_tracks = results_comparison['with_tracking']['total_tracks_created']
    avg_tracking_time = np.mean(results_comparison['with_tracking']['processing_times']) if \
    results_comparison['with_tracking']['processing_times'] else 0

    print(f"\nWITH TRACKING:")
    print(f"  Total Tracks Created: {total_tracks}")
    print(f"  Active Tracks: {tracking_summary.get('active_tracklets', 0)}")
    print(f"  Avg Track Lifetime: {tracking_summary.get('average_lifetime', 0):.1f} frames")
    print(f"  Avg Tracking Processing Time: {avg_tracking_time:.4f}s ({avg_tracking_time * 1000:.2f}ms)")

    # Performance comparison
    if avg_inference_time > 0:
        overhead_ratio = avg_tracking_time / avg_inference_time
        overhead_ms = (avg_tracking_time - avg_inference_time) * 1000
        print(f"  Tracking Overhead: {overhead_ratio:.2f}x ({overhead_ms:+.2f}ms)")

    # Tracking quality metrics
    if 'overall_match_ratio' in tracking_summary:
        print(f"  Overall Match Ratio: {tracking_summary['overall_match_ratio']:.3f}")
        print(f"  Average Match Distance: {tracking_summary['overall_average_distance']:.2f}m")

    # Efficiency metrics
    if total_detections > 0:
        track_efficiency = total_tracks / total_detections
        print(f"  Track Creation Efficiency: {track_efficiency:.3f} (tracks/detections)")

    print("=" * 80)


# Example usage function
def run_processing_example():
    """
    Example function showing how to use the main processing function.
    Replace with your actual parameters.
    """

    # Your model, dataset, and configuration setup
    # net = YourModel()
    # dataset = YourDataset()
    # config = {'data_mode': 'FFT'}  # or 'ADC'
    # checkpoint_filename = "path/to/your/checkpoint.pth"
    # enc = YourEncoder()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run the main processing
    # results = main_processing_with_tracking(
    #     net=net,
    #     dataset=dataset,
    #     config=config,
    #     checkpoint_filename=checkpoint_filename,
    #     enc=enc,
    #     device=device
    # )

    # Access results
    # tracking_analysis = results['tracking_analysis']
    # results_comparison = results['results_comparison']
    # saved_dataframes = results['saved_dataframes']

    pass


if __name__ == "__main__":
    # Replace this with your actual model setup and run
    print("Main processing script loaded. Call main_processing_with_tracking() with your parameters.")
    run_processing_example()