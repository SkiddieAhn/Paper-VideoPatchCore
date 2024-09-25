import torch
import torch.nn as nn
import numpy as np
from functions.pooling import *
import torch.nn.functional as F

resolution = 28
avg = torch.nn.AvgPool2d(3, stride=1)
resize = torch.nn.AdaptiveAvgPool2d(resolution)


resolution2 = 14
avg2 = torch.nn.AvgPool2d(3, stride=1)
resize2 = torch.nn.AdaptiveAvgPool2d(resolution2)


def MakeLocallyAwareFeature(feature_maps, pool_size=64, is_avg=True, len_frame=5):
    '''
    feature_maps[0]: (４*obj, 512, 28, 28)
    feature_maps[1]: (４*obj, 1024, 14, 14)
    output: (obj, 64, ４, 28, 28)
    '''
    if is_avg:
        resized_maps = [resize(avg(fmap)) for fmap in feature_maps] # (４*obj, 512, 28, 28), (４*obj, 1024, 28, 28)
    else:
        resized_maps = [resize(fmap) for fmap in feature_maps] # (４*obj, 512, 28, 28), (４*obj, 1024, 28, 28)
    locally_feature = torch.cat(resized_maps, 1)  # (４*obj, 1536, 28, 28)
    locally_feature = ChannelAvgPool2d(locally_feature, pool_size) # (４*obj, 64, 28, 28)

    chunked_tensors = torch.chunk(locally_feature, chunks=len_frame, dim=0) # ４ * (obj, 64, 28, 28)
    locally_feature = torch.stack(chunked_tensors, dim=0)  # (４, obj, 64, 28, 28)
    locally_feature = locally_feature.permute(1,2,0,3,4) # (obj, 64, ４, 28, 28)

    return locally_feature


def MakeGloballyAwareFeature(feature_maps):
    '''
    feature_maps[0]: (4, 512, 28, 28)
    feature_maps[1]: (4, 512, 14, 14)
    output: (1, 1536, 4)
    '''
    resized_maps = [resize2(avg2(fmap)) for fmap in feature_maps]
    global_feature = torch.cat(resized_maps, 1)
    global_feature = global_feature.permute(1,0,2,3).unsqueeze(0) # (1, 1536, ４, 14, 14)
  
    ap = nn.AvgPool3d((1, 14, 14), stride=(1, 14, 14))
    mp = nn.MaxPool3d((1, 14, 14), stride=(1, 14, 14))

    global_feature = ap(global_feature) + mp(global_feature)
    channel_dim = global_feature.shape[1]
    global_feature = global_feature.view(1, channel_dim, -1)  
  
    return global_feature


def MakeHighlevelFeature(feature):
    '''
    feature: (1, d, 1536)
    output: (d, 1536)
    '''

    pool = nn.MaxPool1d(2, stride=1)
    channels = feature.shape[1]

    # downsampling
    feature0 = feature
    feature1 = pool(feature0) 
    feature2 = pool(feature1) 

    # upsampling + add
    out_feature = F.adaptive_max_pool1d(feature2, output_size=feature1.shape[2])
    out_feature += feature1

    out_feature = F.adaptive_max_pool1d(out_feature, output_size=feature0.shape[2])
    out_feature += feature0

    out_feature = out_feature.view(1, channels, -1).permute(0,2,1)
    highlevel_feature = out_feature.squeeze(0)

    return highlevel_feature



def MakeSpatialFeature(feature, kernel_size=(1,1,1)):
    '''
    visual feature: (obj, 64, 4, 28, 28) -> 
    spatial feature: (obj*28*28, 64)
    '''
    len_channel = feature.shape[1]
    feature1 = TemporalAvgPool3d(feature, 1) # (obj, 64, 1, 28, 28)
    feature2 = TemporalMaxPool3d(feature, 1) # (obj, 64, 1, 28, 28)
    feature = feature1 + feature2

    if kernel_size != (1,1,1):
        feature = AvgPool3d(feature, kernel_size=kernel_size) # (obj, 64, 1, h', w')  

    feature = feature.squeeze(2).permute(0,2,3,1) # (obj, 28, 28, 64)
    object_spatial_feature = feature.reshape(-1, len_channel) # (obj*28*28, 64)
    return object_spatial_feature


def MakeTemporalFeature(feature, mode='train'):
    '''
    visual feature: (obj, 64, 4, 28, 28) -> 
    temporal feature: (obj*4, 64)
    '''
    len_batch = feature.shape[0]
    len_channel = feature.shape[1]
    len_frame = feature.shape[2]
    temporal_feature = torch.zeros((len_batch, len_channel, len_frame-1, resolution, resolution)).to(feature.device)
    for fidx in range(len_frame-1):
        temporal_feature[:,:,fidx,:,:] = feature[:,:,fidx+1,:,:] - feature[:,:,fidx,:,:]

    if mode == 'train':
        temporal_feature = SpatialAvgPool3d(temporal_feature, 1) # (obj, 64, 3, 1, 1)
    else:
        temporal_feature = SpatialMaxPool3d(temporal_feature, 1) # (obj, 64, 3, 1, 1)

    temporal_feature = temporal_feature.permute(0,2,1,3,4) # (obj, 4, 64, 1, 1)
    object_temporal_feature = temporal_feature.reshape(-1, len_channel) # (obj*4, 64)
    return object_temporal_feature
