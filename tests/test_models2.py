import torch
from torchinfo import summary
import yaml

def test_record_carrada():
    dummy_input_rd = torch.rand((1, 1, 5, 256, 64))
    dummy_input_ad = torch.rand((1, 1, 5, 256, 64))
    dummy_input_ra = torch.rand((1, 1, 5, 256, 256))
    from radar_sort.models import MVRecord
    config = yaml.load(open('../configs/carrada/config_mvrecord_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load MV-RECORD ----')
    model = MVRecord(config=backbone_config, n_classes=4, n_frames=5)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=[dummy_input_rd.shape, dummy_input_ra.shape, dummy_input_ad.shape])

    print('---- Test model with a dummy RAD input')
    dummy_output = model(dummy_input_rd.cuda(), dummy_input_ra.cuda(), dummy_input_ad.cuda())
    assert dummy_output[0].shape == (1, 4, 256, 64)
    assert dummy_output[1].shape == (1, 4, 256, 256)

    config = yaml.load(open('../configs/carrada/config_mvrecord_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load MV-RECORD-S ----')
    model = MVRecord(config=backbone_config, n_classes=4, n_frames=5)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=[dummy_input_rd.shape, dummy_input_ra.shape, dummy_input_ad.shape])

    print('---- Test model with a dummy RAD input')
    dummy_output = model(dummy_input_rd.cuda(), dummy_input_ra.cuda(), dummy_input_ad.cuda())
    assert dummy_output[0].shape == (1, 4, 256, 64)
    assert dummy_output[1].shape == (1, 4, 256, 256)

    from radar_sort.models import Record
    config = yaml.load(open('../configs/carrada/config_record_ra_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load RECORD-RA ----')
    model = Record(config=backbone_config, n_class=4, in_channels=1)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=dummy_input_ra.shape)

    print('---- Test model with a dummy RA input')
    dummy_output = model(dummy_input_ra.cuda())
    assert dummy_output.shape == (1, 4, 256, 256)

    config = yaml.load(open('../configs/carrada/config_record_rd_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load RECORD-RD ----')
    model = Record(config=backbone_config, n_class=4, in_channels=1)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=dummy_input_rd.shape)

    print('---- Test model with a dummy RD input')
    dummy_output = model(dummy_input_rd.cuda())
    assert dummy_output.shape == (1, 4, 256, 64)

