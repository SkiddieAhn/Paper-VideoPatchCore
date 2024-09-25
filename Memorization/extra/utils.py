import numpy as np
import argparse
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re


def z_score(arr, eps=1e-8):
    mean = np.mean(arr)
    std_dev = np.std(arr) + eps  # Avoid division by zero
    z_scores = (arr - mean) / std_dev
    return z_scores


def min_max_normalize(arr, eps=1e-8):
    min_val = np.min(arr)
    max_val = np.max(arr)
    denominator = max_val - min_val + eps  # Avoid division by zero

    normalized_arr = (arr - min_val) / denominator
    return normalized_arr


def zero_resizing(cfg, feature, device):
    '''
    object image to centor of zero image
    (objects, 3, 64, 64) -> (objects, 3, 224, 224)
    '''
    b = feature.shape[0]
    c = feature.shape[1]
    zero_tensor = torch.zeros(b, c, cfg.img_size[0], cfg.img_size[1]).to(device)
    start_x = (cfg.img_size[1] - cfg.obj_size[1]) // 2
    start_y = (cfg.img_size[0] - cfg.obj_size[0]) // 2
    end_x = start_x + cfg.obj_size[1]
    end_y = start_y + cfg.obj_size[0]
    zero_tensor[:, :, start_y:end_y, start_x:end_x] = feature
    return zero_tensor


def upsampling(cfg, feature, device):
    '''
    upsampling with bilinear interpolation
    (objects, 3, 64, 64) -> (objects, 3, 224, 224)
    '''
    output = F.interpolate(feature, size=(cfg.img_size[0], cfg.img_size[1]), mode='bilinear', align_corners=True).to(device)
    return output


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))


def setdirs(cfg):
    # train
    train_folders = os.listdir(cfg.train_data)
    train_folders = sorted(train_folders, key=extract_numbers)
    train_folders = [os.path.join(cfg.train_data, aa) for aa in train_folders]
    train_length = len(train_folders)

    # test
    test_folders = os.listdir(cfg.test_data)
    test_folders = sorted(test_folders, key=extract_numbers)
    test_folders = [os.path.join(cfg.test_data, aa) for aa in test_folders]
    test_length = len(test_folders)

    return train_folders, train_length, test_folders, test_length


def makedirs(cfg):
    # local features
    if not os.path.exists(f"{cfg.work_dir}/l_features"):
        os.makedirs(f"{cfg.work_dir}/l_features")

    if not os.path.exists(f"{cfg.work_dir}/l_features/{cfg.dataset}"):
        os.makedirs(f"{cfg.work_dir}/l_features/{cfg.dataset}")

    if not os.path.exists(f"{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}"):
        os.makedirs(f"{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}")

    if not os.path.exists(f"{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/train"):
        os.makedirs(f"{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/train")

    if not os.path.exists(f"{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/test"):
        os.makedirs(f"{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/test")

    # global features
    if not os.path.exists(f"{cfg.work_dir}/g_features"):
        os.makedirs(f"{cfg.work_dir}/g_features")

    if not os.path.exists(f"{cfg.work_dir}/g_features/{cfg.dataset}"):
        os.makedirs(f"{cfg.work_dir}/g_features/{cfg.dataset}")

    if not os.path.exists(f"{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}"):
        os.makedirs(f"{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}")

    if not os.path.exists(f"{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/visual/train"):
        os.makedirs(f"{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/visual/train")

    if not os.path.exists(f"{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/visual/test"):
        os.makedirs(f"{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/visual/test")
