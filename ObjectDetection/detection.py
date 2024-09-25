import torch
from fastprogress import progress_bar
from od_function import *
from extra.utils import makedirs_option
from extra.dataset import dataset_path_for_memory, np_img_load_frame
import cv2
import pickle
import h5py
import numpy as np


class ObjectDetection:
    def __init__(self, cfg, device, train_folders, train_length, test_folders, test_length):
        self.device = device
        self.train_folders = train_folders
        self.train_length = train_length
        self.test_folders = test_folders
        self.test_length = test_length

        # load model
        self.model = get_yolo()

        # save object image
        if cfg.save_image or cfg.save_image_all:
            makedirs_option(cfg, test_folders, 'test')
            # self.save_od_image_w_pickle(cfg, test_folders, test_length, 'test')
            self.save_od_image(cfg, test_folders, test_length, 'test')

        # object detection
        if cfg.train_od:
            self.object_detection(cfg, train_folders, train_length, 'train', cfg.confidence)
            
        if cfg.test_od:
            self.object_detection(cfg, test_folders, test_length, 'test', cfg.test_confidence)

        # save pickle
        if cfg.is_save_train_pickle:
            self.save_pickle(cfg, 'train')
        
        if cfg.is_save_test_pickle:
            self.save_pickle(cfg, 'test')  

        # load pickle
        if cfg.is_load_train_pickle:
            self.load_pickle(cfg, 'train')

        if cfg.is_load_test_pickle:
            self.load_pickle(cfg, 'test')


    def to_numpy(self, tensor):
        tensor = tensor.to('cpu')
        output = np.array(tensor)
        return output


    def save_pickle(self, cfg, mode='train'):
        if mode == 'train':
            data_folders = self.train_folders
            data_length = self.train_length
            confidence = cfg.confidence
        else:
            data_folders = self.test_folders
            data_length = self.test_length
            confidence = cfg.test_confidence

        video_dict={}
        video_dict_path = f'{cfg.work_dir}/objects/{cfg.dataset}/{cfg.dataset}_{mode}.pickle'
        print(f'\nSave {mode} bbox into pickle...\n({video_dict_path})\n')

        for i, folder in progress_bar(enumerate(data_folders), total=data_length):
            one_video = dataset_path_for_memory(cfg, folder)
            folder_name = folder.split('/')[-1].split('.')[0]
            key = str(i+1)
            video_dict[key]=[]

            # one video processing
            for frame_paths in one_video:
                if len(frame_paths) >= 4:
                    standard = len(frame_paths) // 2
                else:
                    standard = -1

                # object detection with last frame
                areas = detect_nob(last_frame=frame_paths[standard],
                                yolo_model=self.model,
                                confidence=confidence,
                                device = self.device,
                                coi = cfg.coi)
                
                areas = get_resized_area(areas, max_w=cfg.max_w, max_h=cfg.max_h, factor_x=cfg.factor_x, factor_y=cfg.factor_y)
                video_dict[key].append(areas)

            # one video ok!
            print(f'[{i+1}] {folder_name}')

        # save bbox
        with open(video_dict_path, 'wb') as f:
            pickle.dump(video_dict, f)


    def load_pickle(self, cfg, mode='train'):
        if mode == 'train':
            data_folders = self.train_folders
            data_length = self.train_length
        else:
            data_folders = self.test_folders
            data_length = self.test_length

        print(f'Save {mode} object batches...')
        video_dict_path = f'{cfg.work_dir}/objects/{cfg.dataset}/{cfg.dataset}_{mode}.pickle'

        with open(video_dict_path, 'rb') as f:
            video_dict = pickle.load(f)

            for i, folder in progress_bar(enumerate(data_folders), total=data_length):
                one_video = dataset_path_for_memory(cfg, folder)
                one_video_batches = []
                one_video_bboxs = video_dict[str(i+1)]
                folder_name = folder.split('/')[-1].split('.')[0]
                file_path = f'{cfg.work_dir}/objects/{cfg.dataset}/{mode}/{folder_name}_object_batches.h5'

                # one video processing
                for j, frame_paths in enumerate(one_video):
                    frame_length = len(frame_paths)

                    areas = one_video_bboxs[j]
                    object_length = len(areas)

                    '''
                    make batch per {input_length} frames
                    '''
                    batch = torch.zeros((object_length, 3, frame_length, cfg.obj_size[0], cfg.obj_size[1])) # (Objects, 3, 4, 64, 64)

                    # save object tensor
                    if len(areas) > 0: 
                        for object_idx, item in enumerate(areas):
                            xmin, ymin, xmax, ymax = int(item[0]), int(item[1]), int(item[2]), int(item[3])

                            for f in range(frame_length):
                                image = cv2.imread(frame_paths[f]) # (C, H, W)
                                cropped_image = image[ymin:ymax, xmin:xmax, :] # (C, obj_H, obj_W)
                                cropped_image = np_img_load_frame(cropped_image, cfg.obj_size[0], cfg.obj_size[1]) # object: (C, 64, 64)
                                cropped_image = torch.from_numpy(cropped_image)
                                batch[object_idx, :, f, :, :] = cropped_image

                        one_video_batches.append(self.to_numpy(batch))
                    
                    else:
                        one_video_batches.append(self.to_numpy(torch.zeros((1))))
            
                # print features 
                print(f'[{i+1}] {folder_name}: {len(one_video_batches)} batches, last_batch.shape:{batch.shape}')
                
                # save features
                with h5py.File(file_path, 'w') as f:
                    for i, v_ft in enumerate(one_video_batches):
                        f.create_dataset(str(i).zfill(4), data=v_ft)  


    def object_detection(self, cfg, data_folders, data_length, mode, confidence):
        print(f'Save {mode} object batches...')

        for i, folder in progress_bar(enumerate(data_folders), total=data_length):
            one_video = dataset_path_for_memory(cfg, folder)
            one_video_batches = []
            folder_name = folder.split('/')[-1].split('.')[0]
            file_path = f'{cfg.work_dir}/objects/{cfg.dataset}/{mode}/{folder_name}_object_batches.h5'

            # one video processing
            for j, frame_paths in enumerate(one_video):
                frame_length = len(frame_paths)
                
                if len(frame_paths) >= 4:
                    standard = len(frame_paths) // 2
                else:
                    standard = -1

                # object detection with last frame
                areas = detect_nob(last_frame=frame_paths[standard],
                                yolo_model=self.model,
                                confidence=confidence,
                                device = self.device,
                                coi = cfg.coi)
                object_length = len(areas)
                
                areas = get_resized_area(areas, max_w=cfg.max_w, max_h=cfg.max_h, factor_x=cfg.factor_x, factor_y=cfg.factor_y)

                '''
                make batch per {input_length} frames
                '''
                batch = torch.zeros((object_length, 3, frame_length, cfg.obj_size[0], cfg.obj_size[1])) # (Objects, 3, 4, 64, 64)

                # save object tensor
                if len(areas) > 0: 
                    for object_idx, item in enumerate(areas):
                        xmin, ymin, xmax, ymax = int(item[0]), int(item[1]), int(item[2]), int(item[3])

                        for f in range(frame_length):
                            image = cv2.imread(frame_paths[f]) # (C, H, W)
                            cropped_image = image[ymin:ymax, xmin:xmax, :] # (C, obj_H, obj_W)
                            cropped_image = np_img_load_frame(cropped_image, cfg.obj_size[0], cfg.obj_size[1]) # object: (C, 64, 64)
                            cropped_image = torch.from_numpy(cropped_image)
                            batch[object_idx, :, f, :, :] = cropped_image

                    one_video_batches.append(self.to_numpy(batch))
                
                else:
                    one_video_batches.append(self.to_numpy(torch.zeros((1))))
            
            # print features 
            print(f'[{i+1}] {folder_name}: {len(one_video_batches)} batches, last_batch.shape:{batch.shape}')
            
            # save features
            with h5py.File(file_path, 'w') as f:
                for i, v_ft in enumerate(one_video_batches):
                    f.create_dataset(str(i).zfill(4), data=v_ft) 


    def save_od_image(self, cfg, data_folders, data_length, mode):
        for i, folder in progress_bar(enumerate(data_folders), total=data_length):
            one_video = dataset_path_for_memory(cfg, folder)
            folder_name = folder.split('/')[-1].split('.')[0]

            # one video processing
            for j, frame_paths in enumerate(one_video):

                if len(frame_paths) >= 4:
                    standard = len(frame_paths) // 2
                else:
                    standard = -1

                # object detection with last frame
                areas = detect_nob(last_frame=frame_paths[standard],
                                yolo_model=self.model,
                                confidence=cfg.test_confidence,
                                device = self.device,
                                coi = cfg.coi)
                
                areas = get_resized_area(areas, max_w=cfg.max_w, max_h=cfg.max_h, factor_x=cfg.factor_x, factor_y=cfg.factor_y)

                # save frame with bbox for specific frame
                if cfg.save_image:
                    image = cv2.imread(frame_paths[standard])

                    for idx, item in enumerate(areas):
                        xmin, ymin, xmax, ymax = int(item[0]), int(item[1]), int(item[2]), int(item[3])

                        cv2.putText(image, str(idx), (xmin,ymin), cv2.FONT_ITALIC, 0.5, (0,0,255), 1)
                        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 1)

                        file_path = f'{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}/{j}clip.png'
                        save_img = (image).astype('uint8')
                        cv2.imwrite(file_path, save_img) 

                # save resized bbox for all image
                if cfg.save_image_all:
                    for k in range(cfg.consecutive):
                        image = cv2.imread(frame_paths[k])
                        for idx, item in enumerate(areas):
                            xmin, ymin, xmax, ymax = int(item[0]), int(item[1]), int(item[2]), int(item[3])
                            cropped_image = image[ymin:ymax, xmin:xmax, :]
                            cropped_image = cv2.resize(cropped_image, (cfg.obj_size[0], cfg.obj_size[1]))

                            file_path = f'{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}_bbox/{j}clip_{idx}object_{k}frame.png'
                            save_img = (cropped_image).astype('uint8')
                            cv2.imwrite(file_path, save_img) 

            print(f'[{i+1}] {folder_name}')



    def save_od_image_w_pickle(self, cfg, data_folders, data_length, mode):
        video_dict_path = f'{cfg.work_dir}/objects/{cfg.dataset}/{cfg.dataset}_{mode}.pickle'

        with open(video_dict_path, 'rb') as f:
            video_dict = pickle.load(f)

            for i, folder in progress_bar(enumerate(data_folders), total=data_length):
                one_video = dataset_path_for_memory(cfg, folder)
                one_video_bboxs = video_dict[str(i+1)]
                folder_name = folder.split('/')[-1].split('.')[0]

                # one video processing
                for j, frame_paths in enumerate(one_video):
                    areas = one_video_bboxs[j]

                    if len(frame_paths) >= 4:
                        standard = len(frame_paths) // 2
                    else:
                        standard = -1

                    # save frame with bbox for specific frame
                    if cfg.save_image:
                        image = cv2.imread(frame_paths[standard])

                        for idx, item in enumerate(areas):
                            xmin, ymin, xmax, ymax = int(item[0]), int(item[1]), int(item[2]), int(item[3])

                            cv2.putText(image, str(idx), (xmin,ymin), cv2.FONT_ITALIC, 0.5, (0,0,255), 1)
                            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 1)

                            file_path = f'{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}/{j}clip.png'
                            save_img = (image).astype('uint8')
                            cv2.imwrite(file_path, save_img) 

                    # save resized bbox for all image
                    if cfg.save_image_all:
                        for k in range(cfg.consecutive):
                            image = cv2.imread(frame_paths[k])
                            for idx, item in enumerate(areas):
                                xmin, ymin, xmax, ymax = int(item[0]), int(item[1]), int(item[2]), int(item[3])
                                cropped_image = image[ymin:ymax, xmin:xmax, :]
                                cropped_image = cv2.resize(cropped_image, (cfg.obj_size[0], cfg.obj_size[1]))

                                file_path = f'{cfg.work_dir}/od_save/{cfg.dataset}/{mode}/{folder_name}_bbox/{j}clip_{idx}object_{k}frame.png'
                                save_img = (cropped_image).astype('uint8')
                                cv2.imwrite(file_path, save_img) 

                print(f'[{i+1}] {folder_name}')

    
