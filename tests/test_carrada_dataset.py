import yaml
import os

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from data.carrada.dataset import Carrada
from data.carrada.dataloaders import SequenceCarradaDataset, CarradaDataset, HFlip, VFlip
from torch.utils.data import DataLoader
from utils.paths_internal import CONFIG_DIR

CONFIG_PTH = CONFIG_DIR
config_file = 'config_mac_mvrecord_carrada.yaml'

def test_carrada_dataset():
    """Method to test the dataset"""
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', config_file), 'r'), yaml.FullLoader)
    dataset = Carrada(config).get('Train')
    assert '2019-09-16-12-55-51' in dataset.keys()
    # assert '2019-09-16-12-52-12' in dataset.keys()
    # assert '2020-02-28-13-05-44' in dataset.keys()


def test_carrada_sequence():
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', config_file), 'r'), yaml.FullLoader)
    dataset = Carrada(config).get('Train')
    dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                            shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        seq_name, seq = data
        if i == 0:
            seq = [subseq[0] for subseq in seq]
            assert seq_name[0] == '2019-09-16-12-52-12'
            assert '000163' in seq
            assert '001015' in seq
        else:
            break


def test_carradadataset():
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', config_file), 'r'), yaml.FullLoader)
    paths = config['dataset_cfg']
    n_frames = 3
    dataset = Carrada(config).get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = os.path.join(paths['carrada'], seq_name[0])
        frame_dataloader = DataLoader(CarradaDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            assert list(frame['rd_matrix'].shape[2:]) == [256, 64]
            assert list(frame['ra_matrix'].shape[2:]) == [256, 256]
            assert list(frame['ad_matrix'].shape[2:]) == [256, 64]
            assert frame['rd_matrix'].shape[1] == n_frames
            assert list(frame['rd_mask'].shape[2:]) == [256, 64]
            assert list(frame['ra_mask'].shape[2:]) == [256, 256]
        break


def test_carrada_subflip():
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', config_file), 'r'), yaml.FullLoader)
    paths = config['dataset_cfg']
    n_frames = 3
    dataset = Carrada(config).get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = os.path.join(paths['carrada'], seq_name[0])
        frame_dataloader = DataLoader(CarradaDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            rd_matrix = frame['rd_matrix'][0].cpu().detach().numpy()
            rd_mask = frame['rd_mask'][0].cpu().detach().numpy()
            rd_frame_test = {'matrix': rd_matrix,
                             'mask': rd_mask}
            rd_frame_vflip = VFlip()(rd_frame_test)
            rd_matrix_vflip = rd_frame_vflip['matrix']
            rd_frame_hflip = HFlip()(rd_frame_test)
            rd_matrix_hflip = rd_frame_hflip['matrix']
            assert rd_matrix[0][0][0] == rd_matrix_vflip[0][0][-1]
            assert rd_matrix[0][0][-1] == rd_matrix_vflip[0][0][0]
            assert rd_matrix[0][0][0] == rd_matrix_hflip[0][-1][0]
            assert rd_matrix[0][-1][0] == rd_matrix_hflip[0][0][0]
        break


def test_visualize_samples():
    """Visualize sample data with transformations to understand structure and augmentations"""
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', config_file), 'r'), yaml.FullLoader)
    paths = config['dataset_cfg']
    n_frames = 500
    target_seq = '2019-09-16-12-55-51'  # Your specific sequence

    class_colors = {
        0: 'black',  # Background
        1: 'red',  # Class 1 (e.g., vehicle)
        2: 'blue',  # Class 2 (e.g., pedestrian)
        3: 'green'  # Class 3 (e.g., cyclist)
    }
    cmap_seg = ListedColormap([class_colors[k] for k in sorted(class_colors.keys())])

    # Get dataset and dataloader
    dataset = Carrada(config).get('Train')

    # Verify sequence exists
    if target_seq not in dataset:
        raise ValueError(f"Sequence {target_seq} not found in dataset!")

    # Create dataloader for just our target sequence
    seq_dataloader = DataLoader(
        SequenceCarradaDataset({target_seq: dataset[target_seq]}),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Get first sequence
    seq_name, seq = next(iter(seq_dataloader))
    path_to_frames = os.path.join(paths['carrada'], seq_name[0])

    # Create dataloader with deterministic transforms for visualization
    frame_dataloader = DataLoader(
        CarradaDataset(seq, 'dense', path_to_frames, process_signal=True, n_frames=n_frames),
        shuffle=False,
        batch_size=1,
        num_workers=0
    )

    # Get sample data
    sample = next(iter(frame_dataloader))

    # Extract first sample from batch
    rd_matrix = sample['rd_matrix'][0].cpu().numpy()  # Shape: [n_frames, H, W]
    rd_mask = sample['rd_mask'][0].cpu().numpy()
    ra_matrix = sample['ra_matrix'][0].cpu().numpy()
    ra_mask = sample['ra_mask'][0].cpu().numpy()

    # Apply transforms manually for visualization
    original_rd = {'matrix': rd_matrix, 'mask': rd_mask}  # 3D array [n_frames, H, W]
    hflip_rd = HFlip()(original_rd.copy())
    vflip_rd = VFlip()(original_rd.copy())

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot Range Doppler data
    plt.subplot(2, 3, 1)
    plt.imshow(original_rd['matrix'][-1].squeeze(), cmap='viridis')
    plt.title('RD Matrix (Original)')

    plt.subplot(2, 3, 2)
    plt.imshow(hflip_rd['matrix'][-1].squeeze(), cmap='viridis')
    plt.title('RD Matrix (HFlip)')

    plt.subplot(2, 3, 3)
    plt.imshow(vflip_rd['matrix'][-1].squeeze(), cmap='viridis')
    plt.title('RD Matrix (VFlip)')

    # Plot Range Angle data
    plt.subplot(2, 3, 4)
    plt.imshow(ra_matrix[-1].squeeze(), cmap='viridis')
    plt.title('RA Matrix (Original)')

    # Plot Masks
    plt.subplot(2, 3, 5)
    # plt.imshow(original_rd['mask'][-1].squeeze(), cmap='gray')
    # plt.title('RD Mask (Original)')
    plt.imshow(original_rd['mask'][-1].squeeze(), cmap=cmap_seg)
    plt.title('RD Mask (Original)')

    plt.subplot(2, 3, 6)
    plt.imshow(ra_mask[-1].squeeze(), cmap='gray')
    plt.title('RA Mask (Original)')

    plt.suptitle(f'Visualization for Sequence: {target_seq}', y=1.02)
    handles = [mpatches.Patch(color=class_colors[k], label=f'Class {k}') for k in sorted(class_colors.keys())]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_carrada_dataset()
    # test_carrada_sequence()
    # test_carradadataset()
    # test_carrada_subflip()
    test_visualize_samples()