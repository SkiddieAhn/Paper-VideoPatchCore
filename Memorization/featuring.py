import torch
from fastprogress import progress_bar
import clip
from functions.feature_func import *
from extra.dataset import dataset_for_memory
from extra.utils import zero_resizing, upsampling
import h5py

'''
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Save features
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
class Featuring:
    def __init__(self, cfg, device, train_folders, train_length, test_folders, test_length):
        self.device = device

        # load model
        self.feature_maps = []
        self.model, _ = clip.load(cfg.clip_model, device=device)
        self.model.visual.layer2[-1].register_forward_hook(self.hook)
        self.model.visual.layer3[-1].register_forward_hook(self.hook)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # save train, test feature
        if cfg.save_memory == True:
            self.featuring_global(cfg, 'train', train_folders, train_length)
            self.featuring_local(cfg, 'train', train_folders, train_length)
        else:
            self.featuring_global(cfg, 'test', test_folders, test_length)
            self.featuring_local(cfg, 'test', test_folders, test_length)


    def hook(self, module, input, output) -> None:
        """This hook saves the extracted feature map on self.featured."""
        if torch.cuda.is_available():
            output = output.type(torch.cuda.FloatTensor)
        else:
            output = output.type(torch.FloatTensor)
        self.feature_maps.append(output)


    def to_numpy(self, tensor):
        tensor = tensor.to('cpu')
        output = np.array(tensor)
        return output


    def featuring_local(self, cfg, mode, data_folders, data_length):
        print(f'Save {mode} local features...')

        with torch.no_grad():
            for i, folder in progress_bar(enumerate(data_folders), total=data_length):
                folder_name = folder.split('/')[-1].split('.')[0]
                video_file_path = f'{cfg.work_dir}/objects/{cfg.dataset}/{mode}/{folder_name}_object_batches.h5'
                train_file_path = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/{mode}/{folder_name}_video_features.h5'
                one_video_features = []

                # make features per video
                with h5py.File(video_file_path, 'r') as one_video:
                    for key in one_video:
                        clip = one_video[str(key)][:] 
                        clip = torch.from_numpy(clip).to(self.device) # (obj, 3, ４, 64, 64)

                        if clip.dim() > 1: 
                            for fidx in range(cfg.consecutive):
                                f_objects = clip[:,:,fidx,:,:].to(self.device) # (objects, 3, 64, 64)
                                if fidx == 0:
                                    frames_objects = f_objects
                                else:
                                    frames_objects = torch.cat([frames_objects, f_objects], dim=0) # -> (４*objects, 3, 64, 64)

                            if (frames_objects.shape[2] != cfg.img_size[0]) or (frames_objects.shape[3] != cfg.img_size[1]):
                                frames_objects = upsampling(cfg, frames_objects, self.device) # (４*objects, 3, 224, 224)

                            self.feature_maps = []
                            _ = self.model.visual(frames_objects) # (４*obj, 512, 28, 28), (４*obj, 1024, 14, 14)
                            locally_feature = MakeLocallyAwareFeature(self.feature_maps, pool_size=cfg.cnl_pool, len_frame=cfg.consecutive) # (obj, 32, ４, 28, 28)
                            one_video_features.append(self.to_numpy(locally_feature))
                        else:
                            one_video_features.append(self.to_numpy(clip))

                    # print features 
                    print(f'[{i+1}] {folder_name}: {len(one_video_features)} features, last_batch.shape:{locally_feature.shape}')
                    
                    # save features
                    with h5py.File(train_file_path, 'w') as f:
                        for i, v_ft in enumerate(one_video_features):
                            f.create_dataset(str(i).zfill(4), data=v_ft)  # (obj, 64, ４, 28, 28)


    def featuring_global(self, cfg, mode, data_folders, data_length):
        print(f'Save {mode} global features...')

        with torch.no_grad():
            for i, folder in progress_bar(enumerate(data_folders), total=data_length):
                one_video = dataset_for_memory(cfg, folder)
                folder_name = folder.split('/')[-1].split('.')[0]
                
                train_file_path = f'{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/visual/{mode}/{folder_name}_video_features.h5'
                one_video_features = []

                # make features per video
                for clip in one_video:
                    for fidx in range(cfg.consecutive):
                        frame = clip[3*fidx:3+3*fidx].unsqueeze(0).to(self.device) # (1, 3, 64, 64)
                        if fidx == 0:
                            frames = frame
                        else:
                            frames = torch.cat([frames, frame], dim=0) # -> (４, 3, 64, 64)

                    self.feature_maps = []
                    _ = self.model.visual(frames) 
                    global_feature = MakeGloballyAwareFeature(self.feature_maps) # (1, 4, 2048)
                    one_video_features.append(self.to_numpy(global_feature))

                # print features 
                print(f'[{i+1}] {folder_name}: {len(one_video_features)} features, last_batch.shape:{global_feature.shape}')
                
                # save features
                with h5py.File(train_file_path, 'w') as f:
                    for i, v_ft in enumerate(one_video_features):
                        f.create_dataset(str(i).zfill(4), data=v_ft) # (1, 4, 2048)
