import os
import json
import argparse
from pathlib import Path

from model.FFTRadNet_ViT import FFTRadNet_ViT
from model.FFTRadNet_ViT import FFTRadNet_ViT_ADC
from dataset.dataset import RADIal
from dataset.encoder_NEW import RAEncoder as ra_encoder

from main_processing_with_tracking import main_processing_with_tracking


def main(config, checkpoint_filename,difficult):
    import torch
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


    # Call the main processing function with your parameters
    results = main_processing_with_tracking(
        net=net,
        dataset=dataset,
        config=config,
        checkpoint_filename=checkpoint_filename,
        enc=enc,
        device=device
    )

    # Access the results
    tracking_analysis = results['tracking_analysis']
    results_comparison = results['results_comparison']
    saved_dataframes = results['saved_dataframes']
    tracklet_manager = results['tracklet_manager']


if __name__=='__main__':

    # path_model_default = '/mnt/data/datasets/radial/gd/models/RADIal_SwinTransformer_ADC.pth'
    path_model_default = '/Volumes/ELEMENTS/datasets/Trained_Models/RADIal_SwinTransformer_ADC.pth'

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