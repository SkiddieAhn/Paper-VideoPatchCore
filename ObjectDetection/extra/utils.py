import numpy as np
import argparse
import os
import re


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
    if not os.path.exists(f"{cfg.work_dir}/objects"):
        os.makedirs(f"{cfg.work_dir}/objects")

    if not os.path.exists(f"{cfg.work_dir}/objects/{cfg.dataset}"):
        os.makedirs(f"{cfg.work_dir}/objects/{cfg.dataset}")

    if not os.path.exists(f"{cfg.work_dir}/objects/{cfg.dataset}/train"):
        os.makedirs(f"{cfg.work_dir}/objects/{cfg.dataset}/train")

    if not os.path.exists(f"{cfg.work_dir}/objects/{cfg.dataset}/test"):
        os.makedirs(f"{cfg.work_dir}/objects/{cfg.dataset}/test")


def makedirs_option(cfg, data_folders, mode):
    for folder in data_folders:
        folder_name = folder.split('/')[-1].split('.')[0]
        if not os.path.exists(f"{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}"):
            os.makedirs(f"{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}")

        if cfg.save_image_all:
            if not os.path.exists(f"{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}_bbox"):
                os.makedirs(f"{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}_bbox")