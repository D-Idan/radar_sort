import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.T_FFTRadNet.RadIal.ADCProcessing.DBReader.DBReader import SyncReader
from utils.T_FFTRadNet.RadIal.ADCProcessing.rpl import RadarSignalProcessing

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080


# Input Images Shape
input_shape = (540, 960)  # (height, width)

# Scale factor for resizing
SCALE_FACTOR = (input_shape[1] / ImageWidth, input_shape[0] / ImageHeight)

def image_scale_factor(image):
    """
    Calculate the scale factor for resizing the image to the input shape.
    """
    height, width = image.shape[:2]
    scale_width = width / ImageWidth
    scale_height = height / ImageHeight
    return scale_width, scale_height

def draw_bbox(image, bbox, label_text):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


def generate_figure(rd_path, ra_path, labels, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes
    for _, row in labels.iterrows():
        bbox = (row['x1_pix'], row['y1_pix'], row['x2_pix'], row['y2_pix'])
        # Scale bounding box coordinates
        scale_factor = image_scale_factor(img)
        bbox = (int(bbox[0] * scale_factor[0]), int(bbox[1] * scale_factor[1]),
                int(bbox[2] * scale_factor[0]), int(bbox[3] * scale_factor[1]))
        label = f"R:{row['radar_R_m']:.1f} A:{row['radar_A_deg']:.1f} D:{row['radar_D_mps']:.1f}"
        img = draw_bbox(img, bbox, label)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Camera Image with BBoxes")
    axs[0, 0].axis('off')

    # # Point Cloud
    # rsp_pc = RadarSignalProcessing(calibration_path, method='PC')
    # pc = rsp_pc.run(sample['radar_ch0']['data'], sample['radar_ch1']['data'], sample['radar_ch2']['data'], sample['radar_ch3']['data'])
    # axs[0, 1].polar(pc[:, 2], pc[:, 0], '.')
    # axs[0, 1].set_title("Bird-Eye View (BEV) - Polar Plot")

    # Range-Doppler
    rd_map = np.load(rd_path)
    axs[1, 0].imshow(rd_map)
    axs[1, 0].set_title("Range-Doppler Map")
    axs[1, 0].axis('off')

    # Range-Azimuth
    ra = np.load(ra_path)
    axs[1, 1].imshow(ra)
    axs[1, 1].set_title("Range-Azimuth Map")
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig


def main(labels_csv, image_dir, rd_dir, ra_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')

    dataset = labels_df['dataset'].unique()

    dataset_path = Path(image_dir).parent
    if not os.path.isdir(dataset_path):
        print(f"Skipping {dataset}, directory not found.")

    sample_ids = labels_df[labels_df['dataset'].isin(dataset)]['numSample'].unique()
    for sample_id in sample_ids:

        labels = labels_df[(labels_df['dataset'].isin(dataset)) & (labels_df['numSample'] == sample_id)]

        image_filename = f"image_{sample_id:06d}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        rd_filename = f"rd_{sample_id:06d}.npy"
        rd_path = os.path.join(rd_dir, rd_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        ra_filename = f"ra_{sample_id:06d}.npy"
        ra_path = os.path.join(ra_dir, ra_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue



        fig = generate_figure(rd_path, ra_path, labels, image_path)
        output_path = os.path.join(output_dir, f"viz_{sample_id:06d}.png")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved visualization to {output_path}")


if __name__ == '__main__':
    import argparse

    path_repo = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils')
    if not path_repo.exists():
        path_repo = Path('/mnt/data/myprojects/PycharmProjects/thesis_repos/radar_sort/utils')

    path_config_default = path_repo / Path('T_FFTRadNet/RadIal/ADCProcessing/data_config.json')

    parser = argparse.ArgumentParser(description='Radar-Camera Visualization Script')
    parser.add_argument('-c', '--config', default=str(path_config_default),type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--output_dir', default=None, help='Directory to save output visualizations')

    args = parser.parse_args()

    config = json.load(open(args.config))

    record = config['target_value']
    root_folder = Path(config['Data_Dir'], 'RadIal_Data',record)
    labels_csv = Path(root_folder, 'labels.csv')
    image_dir = Path(root_folder, 'camera')
    rd_dir = Path(root_folder, 'radar_RD')
    ra_dir = Path(root_folder, 'radar_RA')

    # Prepare output folder structure
    output_dir = ra_dir.parent / 'output_visualizations'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)


    main(labels_csv, image_dir, rd_dir, ra_dir, output_dir)
