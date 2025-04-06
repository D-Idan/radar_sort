# main.py

"""Main script to test a pretrained model"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from radar_nextstop.object_detector import detect_objects
from radar_nextstop.track_manager import TrackManager
from utils.paths_collector import Paths
from data_loader import load_carrada_data, load_carrada_frame_dataloader
from tester import Tester

from mvrss.utils.functions import count_params

from mvrss.models import TMVANet, MVNet
import platform


cfg_mac = "/Users/daniel/Idan/University/Masters/Thesis/2024/datasets/logs/carrada/mvnet/mvnet_e300_lr0.0001_s42_0/config.json"
cfg_ubuntu = "/mnt/data/myprojects/PycharmProjects/thesis_repos/MVRSS/logs/carrada/mvnet/mvnet_e300_lr0.0001_s42_0/config.json"
if platform.system() == 'Darwin':  # macOS
    cfg = cfg_mac
elif platform.system() == 'Linux':  # Ubuntu
    cfg = cfg_ubuntu
else:
    raise EnvironmentError("Unsupported operating system")

target_seq = '2019-09-16-12-55-51'  # None

def test_model():

    # Initialize tracker parameters
    tracker = TrackManager(init_frames_needed=2, max_missed=3)


    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default=cfg)
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

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

        if cfg['model'] == 'mvnet':
            run_result = tester.predict_step(model, frame_dataloader, save_plot_path=None)
        else:
            run_result = tester.predict_step(model, frame_dataloader, save_plot_path=None)

        # Process each frame
        for t in range(len(frame_dataloader)):
            seg_mask_rd = run_result['rd_outputs'][t]
            seg_mask_ra = run_result['ra_outputs'][t]
            rd_frame = run_result['rd_data'][t]
            ra_frame = run_result['ra_data'][t]

            # Detect objects from segmentation mask
            detections_rd = detect_objects(seg_mask_rd, min_area=50)
            detections_ra = detect_objects(seg_mask_ra, min_area=50)

            # # Update tracker with detections; get active tracks
            # active_tracks = tracker.update(detections)
            #
            # # Print active track IDs
            # active_ids = [track.track_id for track in active_tracks]
            # print(f"Frame {t}: {len(detections)} detections, Active track IDs: {active_ids}")
            #
            # # Optional: If radar point data is available, assign points to tracks here
            #
            # # Visualize the RD and RA matrices with bounding boxes from segmentation mask
            # plot_rd_ra_with_bboxes(rd_frame, ra_frame, seg_mask, min_area=50)

        print("Tracking complete.")
    print(f"Finished processing sequence {i + 1}/{len(seq_testloader)}")




if __name__ == '__main__':
    test_model()