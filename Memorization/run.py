import argparse
import torch
from extra.config import update_config
from extra.utils import str2bool, setdirs, makedirs
from featuring import Featuring
from memorizing import SpatialMemorizing, TemporalMemorizing, HighSemanticMemorizing
from inferencing import Inference


def main():
    # set hyper parameters
    parser = argparse.ArgumentParser(description='memorization')
    parser.add_argument('--work_num', default=-1, type=int)
    parser.add_argument('--dataset', default='shanghai', type=str)
    parser.add_argument('--cnl_pool', default=64, type=int, help='channel pooling size for efficient memorizing: avenue: 32, shanghai: 64')
    parser.add_argument('--pool_for_sm', default=4, type=int, help='avg pooling kernel size for spatial memory bank')
    parser.add_argument('--consecutive', default=4, type=int, help='')
    parser.add_argument('--spatial_f_coreset', default=0.01, type=float)
    parser.add_argument('--temporal_f_coreset', default=0.01, type=float)
    parser.add_argument('--highlevel_f_coreset', default=0.01, type=float)
    parser.add_argument('--eps_coreset', default=0.9, type=float)
    parser.add_argument('--random_projection', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--save_feature', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--save_memory', default=True, type=str2bool, nargs='?', const=True)

    args = parser.parse_args()
    cfg = update_config(args, mode='memorization')
    cfg.print_cfg() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    # make and set dirs
    makedirs(cfg)
    train_folders, train_length, test_folders, test_length = setdirs(cfg)

    # save features
    if cfg.save_feature:
        Featuring(cfg, device, train_folders, train_length, test_folders, test_length)

    # save momories
    if cfg.save_memory:
        SpatialMemorizing(cfg, device, train_folders, train_length)
        TemporalMemorizing(cfg, device, train_folders, train_length)
        HighSemanticMemorizing(cfg, device, train_folders, train_length)

    # inference
    Inference(cfg, device, test_folders, test_length)


if __name__=="__main__":
    main()
