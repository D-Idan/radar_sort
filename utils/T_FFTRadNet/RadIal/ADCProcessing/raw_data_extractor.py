import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
from rpl import RadarSignalProcessing
from DBReader.DBReader import SyncReader


def ensure_dirs(base_dir, subdirs):
    for sub in subdirs:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)
    # create labels.csv placeholder
    open(os.path.join(base_dir, 'labels.csv'), 'w').close()


def extract_all(config):
    # Load configuration
    cal_table = config['Calibration']
    labels_df = pd.read_csv(config['label_path'], sep=',')
    output_dir = Path(config['Output_Folder']).parent
    record = config['target_value']
    root_folder = Path(config['Data_Dir'], record)

    # Prepare output folder structure
    base = os.path.join(output_dir, 'RadIal_Data',record)
    subdirs = [
        'ADC_Data', 'camera', 'laser_PCL',
        'radar_FFT', 'radar_Freespace', 'radar_PCL'
    ]
    ensure_dirs(base, subdirs)

    # Initialize readers and processors
    db = SyncReader(root_folder, tolerance=200000, silent=True)
    RSP_PC = RadarSignalProcessing(cal_table, method='PC')
    RSP_RD = RadarSignalProcessing(cal_table, method='RD')
    RSP_RA = RadarSignalProcessing(cal_table, method='RA')

    # Filter labels for this record
    rec_labels = labels_df[labels_df['dataset'] == record]
    collected = []

    for idx in tqdm(rec_labels['index'].unique(), desc="Processing Samples", unit="sample"):
        sample = db.GetSensorData(int(idx))
        numSample = rec_labels[rec_labels['index'] == idx]['numSample'].iloc[0]
        tag = f"{numSample:06d}"

        # 1. ADC_Data (skipped)
        # np.save(os.path.join(base, 'ADC_Data', f'adc_{tag}.npy'), adc_data)

        # 2. camera -> save jpg
        cam = sample['camera']['data']
        img_path = os.path.join(base, 'camera', f'image_{tag}.jpg')
        # Resize to (width=960, height=540)
        resized = cv2.resize(cam, (960, 540))
        cv2.imwrite(img_path, resized)
        # imageio.imwrite(img_path, cam)

        # # 3. laser_PCL -> save as NPY
        # pcl = sample['scala']['data']
        # np.save(os.path.join(base, 'laser_PCL', f'laser_{tag}.npy'), pcl)

        # # 4. radar FFT -> RD spectrums
        # rd = RSP_RD.run(
        #     sample['radar_ch0']['data'], sample['radar_ch1']['data'],
        #     sample['radar_ch2']['data'], sample['radar_ch3']['data']
        # )
        # np.save(os.path.join(base, 'radar_FFT', f'rd_{tag}.npy'), rd)

        # # # 5. radar Freespace -> RA map
        # # ra = RSP_RA.run(
        # #     sample['radar_ch0']['data'], sample['radar_ch1']['data'],
        # #     sample['radar_ch2']['data'], sample['radar_ch3']['data']
        # # )
        # # np.save(os.path.join(base, 'radar_Freespace', f'ra_{tag}.npy'), ra)
        #
        # # 6. radar PCL -> PC point cloud
        # pc = RSP_PC.run(
        #     sample['radar_ch0']['data'], sample['radar_ch1']['data'],
        #     sample['radar_ch2']['data'], sample['radar_ch3']['data']
        # )
        # # save as CSV: Range, Doppler, Azimuth, Elevation
        # pcl_cols = ['Range', 'Doppler', 'Azimuth', 'Elevation']
        # pc_df = pd.DataFrame(pc, columns=pcl_cols)
        # pc_df.to_csv(
        #     os.path.join(base, 'radar_PCL', f'pcl_{tag}.csv'),
        #     index=False
        # )

        # collect label entries
        boxes = rec_labels[rec_labels['index'] == idx]
        boxes_out = boxes.copy()
        boxes_out['filename'] = tag
        collected.append(boxes_out)

    # write aggregated labels.csv
    if collected:
        all_labels = pd.concat(collected, ignore_index=True)
        all_labels.to_csv(os.path.join(base, 'labels.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and organize RadIal data')
    parser.add_argument('-c', '--config', default='./data_config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    extract_all(cfg)
