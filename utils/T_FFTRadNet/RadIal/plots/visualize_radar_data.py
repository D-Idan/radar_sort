import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.T_FFTRadNet.RadIal.ADCProcessing.DBReader.DBReader import SyncReader
from utils.T_FFTRadNet.RadIal.ADCProcessing.rpl import RadarSignalProcessing


def draw_bbox(image, bbox, label_text):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


def generate_figure(sample, labels, image_path, calibration_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes
    for _, row in labels.iterrows():
        bbox = (row['x1_pix'], row['y1_pix'], row['x2_pix'], row['y2_pix'])
        label = f"R:{row['radar_R_m']:.1f} A:{row['radar_A_deg']:.1f} D:{row['radar_D_mps']:.1f}"
        img = draw_bbox(img, bbox, label)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Camera Image with BBoxes")
    axs[0, 0].axis('off')

    # Point Cloud
    rsp_pc = RadarSignalProcessing(calibration_path, method='PC')
    pc = rsp_pc.run(sample['radar_ch0']['data'], sample['radar_ch1']['data'], sample['radar_ch2']['data'], sample['radar_ch3']['data'])
    axs[0, 1].polar(pc[:, 2], pc[:, 0], '.')
    axs[0, 1].set_title("Bird-Eye View (BEV) - Polar Plot")

    # Range-Doppler
    rsp_rd = RadarSignalProcessing(calibration_path, method='RD')
    rd = rsp_rd.run(sample['radar_ch0']['data'], sample['radar_ch1']['data'], sample['radar_ch2']['data'], sample['radar_ch3']['data'])
    rd_map = np.log10(np.sum(np.abs(rd), axis=2) + 1e-6)
    axs[1, 0].imshow(rd_map)
    axs[1, 0].set_title("Range-Doppler Map")
    axs[1, 0].axis('off')

    # Range-Azimuth
    rsp_ra = RadarSignalProcessing(calibration_path, method='RA', device='cuda', lib='CuPy')
    ra = rsp_ra.run(sample['radar_ch0']['data'], sample['radar_ch1']['data'], sample['radar_ch2']['data'], sample['radar_ch3']['data'])
    axs[1, 1].imshow(ra)
    axs[1, 1].set_title("Range-Azimuth Map")
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig


def main(labels_csv, image_dir, adc_dir, calibration_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')

    dataset = labels_df['dataset'].unique()

    dataset_path = Path(adc_dir).parent
    if not os.path.isdir(dataset_path):
        print(f"Skipping {dataset}, directory not found.")

    db = SyncReader(dataset_path)

    sample_ids = labels_df[labels_df['dataset'] == dataset]['numSample'].unique()
    for sample_id in sample_ids:
        labels = labels_df[(labels_df['dataset'] == dataset) & (labels_df['numSample'] == sample_id)]
        image_filename = f"image_{sample_id:06d}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            sample = db.GetSensorData(sample_id)
        except Exception as e:
            print(f"Failed to load sample {sample_id} from {dataset}: {e}")
            continue

        fig = generate_figure(sample, labels, image_path, calibration_path)
        output_path = os.path.join(output_dir, f"viz_{sample_id:06d}.png")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved visualization to {output_path}")


if __name__ == '__main__':
    import argparse

    path_config_default = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils/T_FFTRadNet/RadIal/ADCProcessing/data_config.json')
    path_output_default = Path('/Volumes/ELEMENTS/datasets/radial/visualizations')

    parser = argparse.ArgumentParser(description='Radar-Camera Visualization Script')
    parser.add_argument('-c', '--config', default=str(path_config_default),type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--output_dir', default=path_output_default, help='Directory to save output visualizations')

    args = parser.parse_args()

    config = json.load(open(args.config))

    record = config['target_value']
    root_folder = Path(config['Data_Dir'], 'RadIal_Data',record)
    calibration_path = Path(config['Calibration'])
    labels_csv = Path(root_folder, 'labels.csv')
    image_dir = Path(root_folder, 'camera')
    adc_dir = Path(root_folder, 'ADC_Data')

    # Prepare output folder structure
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)


    main(labels_csv, image_dir, adc_dir, calibration_path, output_dir)
