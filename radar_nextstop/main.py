# main.py

"""Main script to test a pretrained model"""

import argparse
import json
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

from data.carrada.import_utils import paths_npy2jpg
from radar_nextstop.utils_nextstop import plot_rd_ra_with_bboxes, convert_pixel_to_radar_coords

# Get the current file's directory and move up to the project's root
ROOT_DIR = Path(__file__).resolve().parents[1]
# Add the root directory to sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from radar_nextstop.visualize_radar_nextsort import plot_image_2D, plot_image_RGB, visualize_mask_and_bboxes, \
    visualize_radar_nextsort, plot_combined_results
from radar_nextstop.object_detector import detect_objects
from radar_nextstop.track_manager import TrackManager
from utils.paths_collector import Paths
from data_loader import load_carrada_data, load_carrada_frame_dataloader
from tester import Tester

from mvrss.utils.functions import count_params, normalize

from mvrss.models import TMVANet, MVNet
import platform


# cfg_mac = "/Users/daniel/Idan/University/Masters/Thesis/2024/datasets/logs/carrada/mvnet/mvnet_e300_lr0.0001_s42_0/config.json"
cfg_mac = "/Users/daniel/Idan/University/Masters/Thesis/2024/datasets/logs/carrada/tmvanet/tmvanet_e300_lr0.0001_s42_5/config.json"

# cfg_ubuntu = "/mnt/data/myprojects/PycharmProjects/thesis_repos/MVRSS/logs/carrada/mvnet/mvnet_e300_lr0.0001_s42_0/config.json"
cfg_ubuntu = "/mnt/data/myprojects/PycharmProjects/thesis_repos/MVRSS/logs/carrada/tmvanet/tmvanet_e300_lr0.0001_s42_5/config.json"
if platform.system() == 'Darwin':  # macOS
    cfg = cfg_mac
elif platform.system() == 'Linux':  # Ubuntu
    cfg = cfg_ubuntu
else:
    raise EnvironmentError("Unsupported operating system")

target_seq = '2019-09-16-12-55-51'  # None
# target_seq = '2020-02-28-13-06-53'  # None

def test_model(cfg=cfg):

    # Initialize tracker parameters
    tracker = TrackManager(init_frames_needed=2, max_missed=3)


    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default=cfg)
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
        print(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['device'] = device
    cfg['paths'] = Paths().get()

    paths = Paths().get()
    exp_name = cfg['name_exp'] + '_' + str(cfg['version'])
    path = paths['logs'] / cfg['dataset'] / cfg['model'] / exp_name
    model_path = path / 'results' / 'model.pt'
    test_results_path = path / 'results' / 'test_results.json'

    if cfg['model'] == 'mvnet':
        model = MVNet(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    else:
        model = TMVANet(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)

    tester = Tester(cfg)
    seq_testloader = load_carrada_data(cfg, split='Train', target_seq=target_seq, batch_size=1, num_workers=0, shuffle=False)
    tester.set_annot_type(cfg['annot_type'])

    for i, sequence_data in enumerate(seq_testloader):
        frame_dataloader = load_carrada_frame_dataloader(cfg, seq_name=sequence_data[0], seq=sequence_data[1],
                                                          split='Train', add_temp=False)

        # cfg_bb = cfg.copy()
        # cfg_bb['annot_type'] = 'box'
        # frame_dataloader_bb = load_carrada_frame_dataloader(cfg_bb, seq_name=sequence_data[0], seq=sequence_data[1],
        #                                                     split='Train', add_temp=False)
        #
        #
        # for (ind_batch, batch_frames), (_, batch_frames_bb) in zip(enumerate(frame_dataloader), enumerate(frame_dataloader_bb)):

        for ind_batch, batch_frames in enumerate(frame_dataloader):
            if cfg['model'] == 'mvnet':
                run_result = tester.predict_step(
                    model, batch_frames, save_plot_path=None)
            else:
                run_result = tester.predict_step(
                    model, batch_frames, save_plot_path=None)

            # Process each batch frames
            for t in range(len(run_result['rd_outputs'])):
                path_rel_img = paths_npy2jpg(batch_frames['paths_ra'][t])

                seg_pred_rd = normalize(run_result['rd_outputs'][t], signal_type='range_doppler', norm_type='local').cpu()
                seg_pred_ra = normalize(run_result['ra_outputs'][t], signal_type='range_angle', norm_type='local').cpu()
                seg_mask_rd = torch.argmax(seg_pred_rd, dim=0)
                seg_mask_ra = torch.argmax(seg_pred_ra, dim=0)

                gt_ra = torch.argmax(run_result['ra_mask'][t], dim=0).cpu()
                gt_rd = torch.argmax(run_result['rd_mask'][t], dim=0).cpu()

                rd_frame_input = run_result['rd_data'][t].cpu()
                ra_frame_input = run_result['ra_data'][t].cpu()

                if 'flip' in cfg['transformations']:
                    ra_frame_input = torch.flip(ra_frame_input, [1])
                    gt_ra = torch.flip(gt_ra, [0])
                    seg_mask_ra = torch.flip(seg_mask_ra, [0])
                    seg_pred_ra = torch.flip(seg_pred_ra, [1])



                # Detect objects from segmentation mask for RA
                detections_ra = detect_objects(seg_mask_ra, min_area=50)



                # # Update tracker with detections; get active tracks
                # if detections_ra:
                #     detections_rd = detect_objects(seg_mask_rd, min_area=50)
                #     cx = detections_rd[-1].cx
                #     cy = detections_rd[-1].cy
                #     print(convert_pixel_to_radar_coords([cx, cy], 'RD'))
                #
                #
                #     # Convert detections to the format expected by the tracker
                #     active_tracks = tracker.update(detections_ra)
                #
                #     # Print active track IDs
                #     active_ids = [track.track_id for track in active_tracks]
                #     print(f"Frame {t}: {len(detections_ra)} detections, Active track IDs: {active_ids}")
                #
                #     # Optional: If radar point data is available, assign points to tracks here

                if ind_batch > 4 and ind_batch < 12:
                    detections_rd = detect_objects(seg_mask_rd, min_area=50)
                    # Prepare inputs
                    vis_data = visualize_radar_nextsort(
                        rd_data=rd_frame_input,
                        ra_data=ra_frame_input,
                        rd_mask=gt_rd,
                        ra_mask=gt_ra,
                        rd_pred_masks=seg_mask_rd,
                        ra_pred_masks=seg_mask_ra,
                        nb_classes=cfg['nb_classes'],
                        output_path='./results' , # None,
                        camera_image_path=path_rel_img['img_path'],
                        frame_num=path_rel_img['img_num'],
                        frame_num_in_seq=path_rel_img['img_seq'],

                    )

                    # Pass directly to plot_combined_results
                    plot_combined_results(**vis_data,
                                          rd_detections_pred=detections_rd,
                                          ra_detections_pred=detections_ra,
                                          plot_ra_cartesian=False,
                                          )

            print("Tracking complete.")
        print(f"Finished processing sequence {i + 1}/{len(seq_testloader)}")




if __name__ == '__main__':
    test_model()