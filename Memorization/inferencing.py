from extra.dataset import Label_loader
from extra.utils import min_max_normalize, z_score
import torch
from fastprogress import progress_bar
from functions.feature_func import *
from functions.memory_func import predict
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics
import h5py


'''
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Load memories, GT
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
def load_memory(cfg):
    spatial_memory_file_path = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/spatial_memory_bank_{cfg.spatial_f_coreset}.pt'
    temporal_memory_file_path = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/temporal_memory_bank_{cfg.temporal_f_coreset}.pt'
    highlevel_memory_file_path = f'{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/highlevel_memory_bank_{cfg.highlevel_f_coreset}.pt'

    spatial_memory_bank = torch.load(spatial_memory_file_path)
    temporal_memory_bank = torch.load(temporal_memory_file_path)
    highlevel_memory_bank = torch.load(highlevel_memory_file_path)
    return spatial_memory_bank, temporal_memory_bank, highlevel_memory_bank


def load_gt(cfg, test_folders):
    gt_loader = Label_loader(cfg, test_folders)
    return gt_loader


'''
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Inference 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''

class Inference:
    def __init__(self, cfg, device, test_folders, test_length):
        self.device = device
        self.test_folders = test_folders
        self.test_length = test_length
        self.spatial_memory_bank, self.temporal_memory_bank, self.highlevel_memory_bank = load_memory(cfg)

        self.scores_s = []
        self.scores_t = []
        self.scores_h = []
        self.p = cfg.pool_for_sm

        # Get gt labels
        gt_loader = Label_loader(cfg, test_folders)  
        self.gt = gt_loader()

        # inference
        self.inference(cfg)

        # get auc
        self.calc_best_auc()


    def inference(self, cfg):
        with torch.no_grad():
            for i, folder in progress_bar(enumerate(self.test_folders), total=self.test_length):
                folder_name = folder.split('/')[-1].split('.')[0]
                test_file_path_local = f'{cfg.work_dir}/l_features/{cfg.dataset}/{cfg.cnl_pool}/visual/test/{folder_name}_video_features.h5'
                test_file_path_global =  f'{cfg.work_dir}/g_features/{cfg.dataset}/{cfg.cnl_pool}/visual/test/{folder_name}_video_features.h5'
               
                video_score_s = []
                video_score_t = []
                video_score_h = []

                # one video (local feature)
                with h5py.File(test_file_path_local, 'r') as v_features:
                    for key in v_features:
                        v_ft = v_features[str(key)][:]
                        v_ft = torch.from_numpy(v_ft).to(self.device) # (obj, 64, ４, 28, 28)

                        max_one_score_s = 0
                        max_one_score_t = 0

                        len_object = v_ft.shape[0]

                        # one clip
                        if v_ft.dim() > 1:
                            for obj_idx in range(len_object):
                                obj_ft = v_ft[obj_idx, :, :, :, :].unsqueeze(0) # (1, 64, 4, h', w')
                                spatial_feature = MakeSpatialFeature(obj_ft, kernel_size=(1, self.p, self.p)) # (h'w', 64)
                                temporal_feature = MakeTemporalFeature(obj_ft, mode='test') # (d', 64)

                                one_score_s = predict(spatial_feature, self.spatial_memory_bank, cfg.power_n).cpu()
                                one_score_t = predict(temporal_feature, self.temporal_memory_bank, cfg.power_n).cpu()

                                max_one_score_s = max(max_one_score_s, one_score_s)
                                max_one_score_t = max(max_one_score_t, one_score_t)

                        for _ in range(cfg.consecutive):
                            video_score_s.append(max_one_score_s)
                            video_score_t.append(max_one_score_t)

                # make scores
                self.scores_s.append(video_score_s)
                self.scores_t.append(video_score_t)

                # one video (global feature)
                with h5py.File(test_file_path_global, 'r') as v_features:
                    for key in v_features:
                        v_ft = v_features[str(key)][:]
                        v_ft = torch.from_numpy(v_ft).to(self.device) # (1, 2048, d)

                        # calc anomaly score        
                        highlevel_feature = MakeHighlevelFeature(v_ft) # (d, 2048)

                        one_score_h = predict(highlevel_feature, self.highlevel_memory_bank, cfg.power_n).cpu().detach().numpy()
                        for _ in range(cfg.consecutive):
                            video_score_h.append(one_score_h)

                # make scores
                self.scores_h.append(video_score_h)

                # one video ok!
                print(i, folder_name, end=' \n')


    def calc_best_auc(self):
        best_auc1 = 0 
        best_auc2 = 0 

        #=======================================#
        #           local stream                # 
        #=======================================#

        for sig in range(0,20):
            for num in np.arange(0.0, 1.1, 0.1):
                scores = np.array([], dtype=np.float32)
                labels = np.array([], dtype=np.int8)

                a1 = round(1-num, 1)
                a2 = round(1-a1, 1)

                for i in range(self.test_length):
                    score = a1*np.array(self.scores_s[i]) + a2*np.array(self.scores_t[i])
                    scores = np.concatenate((scores, score), axis=0)

                    label = self.gt[i][:len(score)]
                    labels = np.concatenate((labels, label), axis=0)

                # gaussian filtering
                if sig > 0:
                    scores = gaussian_filter1d(scores, sigma=sig)
                scores = min_max_normalize(scores)

                fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
                auc = metrics.auc(fpr, tpr)  

                # find best model
                if auc > best_auc1:
                    best_auc1 = auc
                    best_a1 = a1
                    best_a2 = a2
                    best_sig = sig

        print('\nspatial, temporal, sigma:', best_a1, best_a2, best_sig)
        print('best_auc_local:', best_auc1)

        #=======================================#
        #           All stream                  # 
        #=======================================#

        for sig in range(0,20):
            for num in np.arange(0.0, 1.1, 0.1):
                scores1 = np.array([], dtype=np.float32)
                scores2 = np.array([], dtype=np.float32)
                scores = np.array([], dtype=np.float32)

                b1 = round(1-num, 1)
                b2 = round(1-b1, 1)

                for i in range(self.test_length):
                    score1 = best_a1*np.array(self.scores_s[i]) + best_a2*np.array(self.scores_t[i])
                    scores1 = np.concatenate((scores1, score1), axis=0)

                    score2 = self.scores_h[i]
                    score2 = z_score(score2)
                    if sig > 0:
                        score2 = gaussian_filter1d(score2, sigma=sig)
                    scores2 = np.concatenate((scores2, score2), axis=0)

                scores1 = z_score(scores1)
                if sig > 0: 
                    scores1 = gaussian_filter1d(scores1, sigma=sig)

                # anomaly score fusion 
                scores = min_max_normalize(b1*scores1 + b2*scores2)
                fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
                auc = metrics.auc(fpr, tpr)  

                # find best model
                if auc > best_auc2:
                    best_auc2 = auc
                    best_b1 = b1
                    best_b2 = b2
                    best_sig = sig

        print('\nlocal, global, sigma:', best_b1, best_b2, best_sig)
        print('best_auc_total:', best_auc2)

