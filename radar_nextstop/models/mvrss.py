from mvrss.utils.functions import normalize


def mvrss_run_net(model, net, rd_data, ra_data, ad_data=None, device='cuda', norm_type='minmax'):

    if model == 'tmvanet':
        ad_data = normalize(ad_data, 'angle_doppler', norm_type=norm_type)
        rd_data = rd_data.unsqueeze(1).float()  # [6,1,5,256,64]
        ra_data = ra_data.unsqueeze(1).float()  # [6,1,5,256,256]
        ad_data = ad_data.unsqueeze(1).float()  # [6,1,5,256,64]
        rd_outputs, ra_outputs = net(rd_data, ra_data, ad_data)
    else:
        rd_outputs, ra_outputs = net(rd_data, ra_data)
    return rd_outputs, ra_outputs


