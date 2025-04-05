# main.py

"""Main script to test a pretrained model"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from utils.paths_collector import Paths
from data_loader import load_carrada_data
from tester import Tester

from mvrss.utils.functions import count_params

from mvrss.models import TMVANet, MVNet


cfg_mac = "/Users/daniel/Idan/University/Masters/Thesis/2024/datasets/logs/carrada/mvnet/mvnet_e300_lr0.0001_s42_0/config.json"
target_seq = '2019-09-16-12-55-51' # None

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default=cfg_mac)
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['device'] = device

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

    if cfg['model'] == 'mvnet':
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=False)
    else:
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=True)
    tester.write_params(test_results_path)

if __name__ == '__main__':
    test_model()