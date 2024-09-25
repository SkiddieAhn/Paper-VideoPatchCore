import os
import torch
import glob
import cv2
import numpy as np
import scipy.io as scio
import re


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))


class dataset_for_memory:
    def __init__(self, cfg, video_folder):
        self.img_h = cfg.img_size[0]
        self.img_w = cfg.img_size[1]
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
        video_clips = []
        curr = indice * (self.clip_length)
        if self.first_no_use:
            curr += 1

        for frame_id in range(curr, curr + self.clip_length):
            video_clips.append(np_load_frame(self.imgs[frame_id], self.img_h, self.img_w))

        video_clips = np.array(video_clips).reshape((-1, self.img_h, self.img_w))
        video_clips = torch.from_numpy(video_clips) # new code
        return video_clips


class Label_loader:
    def __init__(self, cfg, video_folders):
        assert cfg.dataset in ('avenue', 'shanghai', 'iitb'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.cfg = cfg
        self.name = cfg.dataset
        self.frame_path = cfg.test_data
        self.mat_path = f'{cfg.data_root + self.name}/{self.name}.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghai':
            gt = self.load_shanghaitech(self.name)
        elif self.name == 'iitb':
            gt = self.load_iitb()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_shanghaitech(self, name):
        np_list = glob.glob(f'{self.cfg.data_root}/{name}/testframemask/*_*.npy')
        np_list = sorted(np_list, key=extract_numbers)

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt
    
    def load_iitb(self):
        np_list = glob.glob(f'{self.cfg.data_root}/iitb/groundtruth/*')
        np_name_list = os.listdir(f'{self.cfg.data_root}/iitb/groundtruth')

        for i in range(len(np_list)):
            np_list[i] += f'/{np_name_list[i]}.npy'
        np_list = sorted(np_list, key=extract_numbers)

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt


    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt)

        return all_gt