import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from model.FFTRadNet_ViT import FFTRadNet_ViT
from model.FFTRadNet_ViT import FFTRadNet_ViT_ADC
from dataset.dataset import RADIal
# from dataset.encoder import ra_encoder
from dataset.encoder_NEW import RAEncoder as ra_encoder
import cv2
from utils.util import DisplayHMI

def main(config, checkpoint_filename,difficult):

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    if config['data_mode'] != 'ADC':
        net = FFTRadNet_ViT(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

        dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=True,perform_FFT=config['data_mode'])

    else:
        net = FFTRadNet_ViT_ADC(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

        dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=True,perform_FFT='ADC')

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, weights_only=False, map_location=torch.device(device))
    net.load_state_dict(dict['net_state_dict'])
    net = net.double()
    net.eval()


    for data in dataset:
        # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        inputs = torch.tensor(data[0]).permute(2,0,1).to(device).unsqueeze(0)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            if config['data_mode'] == 'ADC':
                intermediate = net.DFT(inputs).detach().cpu().numpy()[0]

            else:
                intermediate = None

        if data[4] is not None: # there is image
            hmi = DisplayHMI(data[4], data[0],outputs,enc,config,intermediate)

            cv2.imshow('FFTRadNet',hmi)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # break
    cv2.destroyAllWindows()


if __name__=='__main__':

    path_model_default = Path('/Volumes/ELEMENTS/datasets/Trained_Models/RADIal_SwinTransformer_ADC.pth')
    if not path_model_default.exists():
        path_model_default = Path('/mnt/data/datasets/radial/gd/models/RADIal_SwinTransformer_ADC.pth')

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='./config/ADC_config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=path_model_default, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult)
