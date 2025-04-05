from mvrss.utils.functions import normalize


def mvrss_run_net(model, net, rd_data, ra_data, ad_data=None, device='cuda', norm_type='minmax'):

    if model == 'tmvanet':
        ad_data = normalize(ad_data, 'angle_doppler', norm_type=norm_type)
        rd_outputs, ra_outputs = net(rd_data, ra_data, ad_data)
    else:
        rd_outputs, ra_outputs = net(rd_data, ra_data)
    return rd_outputs, ra_outputs


