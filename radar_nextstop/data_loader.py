# data_loader.py  This module loads radar data (e.g. range-Doppler (RD), range-Azimuth (RA) matrices, segmentation masks, and point series).
# data_loader.py
import torch
import json
from torch.utils.data import DataLoader
from data.carrada.dataset import Carrada
from mvrss.loaders.dataloaders import SequenceCarradaDataset, CarradaDataset
from mvrss.utils.functions import get_transformations


def load_carrada_data(cfg, split='Train', target_seq=None, batch_size=1, num_workers=0, shuffle=False):
    """Loads and preprocesses radar data for tracking"""
    data = Carrada(config_model=cfg)
    dataset_dict = data.get(split)

    if target_seq is not None:
        dataset_dict = {target_seq: dataset_dict[target_seq]}

    seq_dataset = SequenceCarradaDataset(dataset_dict)

    return DataLoader(
        seq_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def load_carrada_frame_dataloader(cfg, seq_name, seq, split='test', add_temp=False):


    transformations = get_transformations(cfg['transformations'].split(','), split=split,
                                          sizes=(cfg['w_size'], cfg['h_size']))
    path_to_frames = cfg['paths']['carrada'] / seq_name[0]
    return DataLoader(CarradaDataset(seq,
                                     cfg['annot_type'],
                                     path_to_frames,
                                     cfg['process_signal'],
                                     cfg['nb_input_channels'],
                                     transformations,
                                     add_temp),
                      shuffle=False,
                      batch_size=cfg['batch_size'],
                      num_workers=4)































