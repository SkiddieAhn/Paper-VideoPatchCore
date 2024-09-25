import os
import torch
import glob
import cv2
import numpy as np
import scipy.io as scio
from PIL import Image
from torchvision import transforms
import random
import re


def np_img_load_frame(img, resize_h, resize_w):
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))


class dataset_path_for_memory:
    def __init__(self, cfg, video_folder):
        self.clip_length = cfg.consecutive
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs = sorted(self.imgs, key=extract_numbers)

        if cfg.dataset == 'iitb':
            self.first_no_use = True
        else:
            self.first_no_use = False

    def __len__(self):
        if self.first_no_use:
            return (len(self.imgs)-1) // self.clip_length
        else:
            return len(self.imgs) // self.clip_length  
        
    def __getitem__(self, indice):
        image_paths = []
        curr = indice * (self.clip_length)
        if self.first_no_use:
            curr += 1

        for frame_id in range(curr, curr + self.clip_length):
            image_paths.append(self.imgs[frame_id])
        return image_paths




