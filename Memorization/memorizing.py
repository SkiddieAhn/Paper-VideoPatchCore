import torch
from fastprogress import progress_bar
from functions.memory_func import get_coreset
from functions.feature_func import *
import h5py


'''
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Save Spaital Memory Bank
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
class SpatialMemorizing:
    def __init__(self, cfg, device, train_folders, train_length):
        self.device = device
        self.train_folders = train_folders
        self.train_length = train_length

        print(cfg.dataset)
        print('Spatial partition...')

        self.p = cfg.pool_for_sm
        self.spatial_memorizing(cfg)


    def spatial_memorizing(self, cfg):
        spatial_memory_bank = []

        with torch.no_grad():
            for i, folder in progress_bar(enumerate(self.train_folders), total=self.train_length):
                folder_name = folder.split('/')[-1].split('.')[0]
                train_file_path = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/train/{folder_name}_video_features.h5'
                one_spatial_memory_bank = []

                # load one video features
                with h5py.File(train_file_path, 'r') as v_features:
                    for key in v_features:
                        v_ft = v_features[str(key)][:]
                        v_ft = torch.from_numpy(v_ft).to(self.device) # (obj, 64, ４, 28, 28)
                
                        # make sptaial features and move to memory bank
                        if v_ft.dim() > 1:
                            spatial_feature = MakeSpatialFeature(v_ft, kernel_size=(1, self.p, self.p)) # (obj*h'w', 64)
                            one_spatial_memory_bank.append(spatial_feature)     

                if len(one_spatial_memory_bank) > 0:
                    one_spatial_memory_bank = torch.cat(one_spatial_memory_bank, 0)

                    # Coreset Subsampling
                    if cfg.spatial_f_coreset < 1:
                        print(f'[Spatial] {i+1}/{self.train_length} video -> start coreset subsampling...')
                        one_spatial_coreset_idx = get_coreset(
                            one_spatial_memory_bank.cpu(),
                            l = int(cfg.spatial_f_coreset * one_spatial_memory_bank.shape[0]),
                            eps = cfg.eps_coreset,
                            r_proj = cfg.random_projection,
                            device = self.device
                        )
                        one_spatial_memory_bank = one_spatial_memory_bank[one_spatial_coreset_idx]

                    # stack to memory bank
                    if len(spatial_memory_bank) == 0:
                        spatial_memory_bank = one_spatial_memory_bank
                    else:
                        spatial_memory_bank = torch.cat((spatial_memory_bank, one_spatial_memory_bank), 0)

            # print -> save memory
            print(f'Spatial Memory Bank: {spatial_memory_bank.shape}')
            spatial_memory_file_path = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/spatial_memory_bank_{cfg.spatial_f_coreset}.pt'
            torch.save(spatial_memory_bank, spatial_memory_file_path)


'''
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Save Temporal Memory Bank
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
class TemporalMemorizing:
    def __init__(self, cfg, device, train_folders, train_length):
        self.device = device
        self.train_folders = train_folders
        self.train_length = train_length

        print(cfg.dataset)
        print('Temporal partition...')

        self.temporal_memorizing(cfg)


    def temporal_memorizing(self, cfg):
        temporal_memory_bank = []

        with torch.no_grad():
            for i, folder in progress_bar(enumerate(self.train_folders), total=self.train_length):
                folder_name = folder.split('/')[-1].split('.')[0]
                train_file_path = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/train/{folder_name}_video_features.h5'
                one_temporal_memory_bank = []

               # load one video features
                with h5py.File(train_file_path, 'r') as v_features:
                    for key in v_features:
                        v_ft = v_features[str(key)][:]
                        v_ft = torch.from_numpy(v_ft).to(self.device) # (obj, 64, ４, 28, 28)
                
                        # make temporal features and move to memory bank
                        if v_ft.dim() > 1:
                            temporal_feature = MakeTemporalFeature(v_ft) # (obj*d', 64)
                            one_temporal_memory_bank.append(temporal_feature)

                if len(one_temporal_memory_bank) > 0:
                    one_temporal_memory_bank = torch.cat(one_temporal_memory_bank, 0)

                    # Coreset subsampling
                    if cfg.temporal_f_coreset < 1:
                        print(f'[Temporal] {i+1}/{self.train_length} video -> start coreset subsampling...')
                        one_temporal_coreset_idx = get_coreset(
                            one_temporal_memory_bank.cpu(),
                            l = int(cfg.temporal_f_coreset * one_temporal_memory_bank.shape[0]),
                            eps = cfg.eps_coreset,
                            r_proj = cfg.random_projection,
                            device = self.device
                        )
                        one_temporal_memory_bank = one_temporal_memory_bank[one_temporal_coreset_idx]

                    # stack to memory bank
                    if len(temporal_memory_bank) == 0:
                        temporal_memory_bank = one_temporal_memory_bank
                    else:
                        temporal_memory_bank = torch.cat((temporal_memory_bank, one_temporal_memory_bank), 0)

            # print -> save memory 
            print(f'Temporal Memory Bank: {temporal_memory_bank.shape}')
            temporal_memory_file_path = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/temporal_memory_bank_{cfg.temporal_f_coreset}.pt'
            torch.save(temporal_memory_bank, temporal_memory_file_path)


'''
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Save High-level Memory Bank
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
class HighSemanticMemorizing:
    def __init__(self, cfg, device, train_folders, train_length):
        self.device = device
        self.train_folders = train_folders
        self.train_length = train_length

        print(cfg.dataset)
        print('High-level partition...')

        self.high_level_memorizing(cfg)


    def high_level_memorizing(self, cfg):
        highlevel_memory_bank = []

        with torch.no_grad():
            for i, folder in progress_bar(enumerate(self.train_folders), total=self.train_length):
                folder_name = folder.split('/')[-1].split('.')[0]
                train_file_path = f'{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/visual/train/{folder_name}_video_features.h5'            
                one_highlevel_memory_bank = []

               # load one video features
                with h5py.File(train_file_path, 'r') as v_features:
                    for key in v_features:
                        v_ft = v_features[str(key)][:]
                        v_ft = torch.from_numpy(v_ft).to(self.device) 
                
                        # make high-level features and move to memory bank
                        if v_ft.dim() > 1:
                            highlevel_feature = MakeHighlevelFeature(feature=v_ft) 
                            one_highlevel_memory_bank.append(highlevel_feature)


                if len(one_highlevel_memory_bank) > 0:
                    one_highlevel_memory_bank = torch.cat(one_highlevel_memory_bank, 0)

                    # Coreset subsampling
                    if cfg.highlevel_f_coreset < 1:
                        print(f'[Highlevel] {i+1}/{self.train_length} video -> start coreset subsampling...')
                        one_highlevel_coreset_idx = get_coreset(
                            one_highlevel_memory_bank.cpu(),
                            l = int(cfg.highlevel_f_coreset * one_highlevel_memory_bank.shape[0]),
                            eps = cfg.eps_coreset,
                            r_proj = cfg.random_projection,
                            device = self.device
                        )
                        one_highlevel_memory_bank = one_highlevel_memory_bank[one_highlevel_coreset_idx]

                    # stack to memory bank
                    if len(highlevel_memory_bank) == 0:
                        highlevel_memory_bank = one_highlevel_memory_bank
                    else:
                        highlevel_memory_bank = torch.cat((highlevel_memory_bank, one_highlevel_memory_bank), 0)

            # print -> save memory 
            print(f'Highlevel Memory Bank: {highlevel_memory_bank.shape}')
            highlevel_memory_file_path = f'{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/highlevel_memory_bank_{cfg.highlevel_f_coreset}.pt'
            torch.save(highlevel_memory_bank, highlevel_memory_file_path)