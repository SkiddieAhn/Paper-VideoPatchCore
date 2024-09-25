import torch
import torch.nn as nn

def ChannelAvgPool2d(tensor, out_channel=2):
    # Input: [b, in_c, h, w]
    # Output: [b, out_c, h, w]
    if tensor.shape[1] % out_channel == 0:
        chunked_tensors = torch.chunk(tensor, chunks=out_channel, dim=1)
        pooled_tensor = torch.cat([torch.mean(chunk, dim=1, keepdim=True) for chunk in chunked_tensors], dim=1)
        return pooled_tensor
    else:
        print(f"Error: can't divide {tensor.shape[1]} to {out_channel}")
        return tensor 
    

def ChannelMaxPool2d(tensor, out_channel=2):
    # Input: [b, in_c, h, w]
    # Output: [b, out_c, h, w]
    if tensor.shape[1] % out_channel == 0:
        chunked_tensors = torch.chunk(tensor, chunks=out_channel, dim=1)
        pooled_tensor = torch.cat([torch.max(chunk, dim=1, keepdim=True).values for chunk in chunked_tensors], dim=1)
        return pooled_tensor
    else:
        print(f"Error: can't divide {tensor.shape[1]} to {out_channel}")
        return tensor 
    

def ChannelMaxPool3d(tensor, out_channel=2):
    # Input: [b, in_c, d, h, w]
    # Output: [b, out_c, d, h, w]
    chunked_tensors = torch.chunk(tensor, chunks=out_channel, dim=1)
    pooled_tensor = torch.cat([torch.max(chunk, dim=1, keepdim=True).values for chunk in chunked_tensors], dim=1)
    return pooled_tensor


def ChannelAvgPool3d(tensor, out_channel=2):
    # Input: [b, in_c, d, h, w]
    # Output: [b, out_c, d, h, w]
    chunked_tensors = torch.chunk(tensor, chunks=out_channel, dim=1)
    pooled_tensor = torch.cat([torch.mean(chunk, dim=1, keepdim=True) for chunk in chunked_tensors], dim=1)
    return pooled_tensor
    

def AvgPool3d(tensor, kernel_size=(1, 1, 1)):
    if kernel_size == (1, 1, 1):
        return tensor
    else:
        d = tensor.shape[2] // kernel_size[0]
        h = tensor.shape[3] // kernel_size[1]
        w = tensor.shape[4] // kernel_size[2]
        pool = torch.nn.AdaptiveAvgPool3d((d, h, w))
        pooled_tensor = pool(tensor)
        return pooled_tensor


def MaxPool3d(tensor, kernel_size=(1, 1, 1)):
    if kernel_size == (1, 1, 1):
        return tensor
    else:
        d = tensor.shape[2] // kernel_size[0]
        h = tensor.shape[3] // kernel_size[1]
        w = tensor.shape[4] // kernel_size[2]
        pool = torch.nn.AdaptiveMaxPool3d((d, h, w))
        pooled_tensor = pool(tensor)
        return pooled_tensor


def TemporalAvgPool3d(tensor, out_depths=1):
    # Input: [b, c, in_d, h, w]
    # Output: [b, c, in_d, h, w]
    if tensor.shape[2] % out_depths == 0:
        chunked_tensors = torch.chunk(tensor, chunks=out_depths, dim=2)
        pooled_tensor = torch.cat([torch.mean(chunk, dim=2, keepdim=True) for chunk in chunked_tensors], dim=2)
        return pooled_tensor
    else:
        print(f"Error: can't divide {tensor.shape[2]} to {out_depths}")
        return tensor 


def TemporalMaxPool3d(tensor, out_depths=1):
    # Input: [b, c, in_d, h, w]
    # Output: [b, c, in_d, h, w]
    if tensor.shape[2] % out_depths == 0:
        chunked_tensors = torch.chunk(tensor, chunks=out_depths, dim=2)
        pooled_tensor = torch.cat([torch.max(chunk, dim=2, keepdim=True).values for chunk in chunked_tensors], dim=2)
        return pooled_tensor
    else:
        print(f"Error: can't divide {tensor.shape[2]} to {out_depths}")
        return tensor 
    

def SpatialAvgPool3d(tensor, out_hw=1):
    # Input: [b, c, d, in_h, in_w]
    # Output: [b, c, d, out_h, out_w]
    if tensor.shape[3] % out_hw == 0:
        d = tensor.shape[2]
        pooling = nn.AdaptiveAvgPool3d((d, out_hw, out_hw))
        pooled_tensor = pooling(tensor)
        return pooled_tensor
    else:
        print(f"Error: can't divide {tensor.shape[3]} to {out_hw}")
        return tensor 
    

def SpatialMaxPool3d(tensor, out_hw=1):
    # Input: [b, c, d, in_h, in_w]
    # Output: [b, c, d, out_h, out_w]
    if tensor.shape[3] % out_hw == 0:
        d = tensor.shape[2]
        pooling = nn.AdaptiveMaxPool3d((d, out_hw, out_hw))
        pooled_tensor = pooling(tensor)
        return pooled_tensor
    else:
        print(f"Error: can't divide {tensor.shape[3]} to {out_hw}")
        return tensor 