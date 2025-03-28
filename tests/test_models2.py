# tests/test_record_models.py
import pytest
import torch
import yaml
from torchsummary import summary

from utils.paths_internal import CONFIG_DIR


def test_record_models():
    """Test all RECORD model variants with dummy inputs"""
    # Test config paths - update these based on your project structure
    CONFIG_PATHS = {
        'mvrecord': CONFIG_DIR / 'carrada/config_mac_mvrecord_carrada.yaml',
        # 'mvrecord': 'configs/carrada/config_mvrecord_carrada.yaml',
        # 'record_ra': 'configs/carrada/config_record_ra_carrada.yaml',
        # 'record_rd': 'configs/carrada/config_record_rd_carrada.yaml'
    }

    # Create dummy inputs for different views
    dummy_inputs = {
        'rd': torch.rand((1, 1, 5, 256, 64)),  # [batch, channels, frames, H, W]
        'ra': torch.rand((1, 1, 5, 256, 256)),
        'ad': torch.rand((1, 1, 5, 256, 64))
    }

    # Test MV-RECORD
    config = yaml.load(open(CONFIG_PATHS['mvrecord'], 'r'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'r'), yaml.FullLoader)

    print("\nTesting MV-RECORD...")
    from models.segmentation.record import MVRecord
    model = MVRecord(config=backbone_config, n_classes=4, n_frames=5)

    # Test forward pass
    outputs = model(dummy_inputs['rd'], dummy_inputs['ra'], dummy_inputs['ad'])
    assert outputs[0].shape == (1, 4, 256, 64), "RD output shape mismatch"
    assert outputs[1].shape == (1, 4, 256, 256), "RA output shape mismatch"

    # Test RECORD-RA
    print("\nTesting RECORD-RA...")
    config = yaml.load(open(CONFIG_PATHS['record_ra'], 'r'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'r'), yaml.FullLoader)

    from models.segmentation.record import Record
    model_ra = Record(config=backbone_config, n_class=4, in_channels=1)
    ra_output = model_ra(dummy_inputs['ra'])
    assert ra_output.shape == (1, 4, 256, 256), "RA-only output shape mismatch"

    # Test RECORD-RD
    print("\nTesting RECORD-RD...")
    config = yaml.load(open(CONFIG_PATHS['record_rd'], 'r'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'r'), yaml.FullLoader)

    model_rd = Record(config=backbone_config, n_class=4, in_channels=1)
    rd_output = model_rd(dummy_inputs['rd'])
    assert rd_output.shape == (1, 4, 256, 64), "RD-only output shape mismatch"

    # Test model device compatibility
    if torch.cuda.is_available():
        print("\nTesting GPU compatibility...")
        device = torch.device("cuda:0")

        # Test MV-RECORD on GPU
        model.cuda()
        outputs_gpu = model(dummy_inputs['rd'].cuda(),
                            dummy_inputs['ra'].cuda(),
                            dummy_inputs['ad'].cuda())
        assert outputs_gpu[0].device.type == 'cuda'
        assert outputs_gpu[1].device.type == 'cuda'

        # Test RECORD-RA on GPU
        model_ra.cuda()
        ra_output_gpu = model_ra(dummy_inputs['ra'].cuda())
        assert ra_output_gpu.device.type == 'cuda'

    print("\nAll basic model tests passed!")


if __name__ == "__main__":
    test_record_models()