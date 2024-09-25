import argparse
import torch
from extra.config import update_config
from extra.utils import str2bool, setdirs, makedirs
from detection import ObjectDetection

def main():
    # set hyper parameters
    parser = argparse.ArgumentParser(description='object_detection')
    parser.add_argument('--work_num', default=-1, type=int)
    parser.add_argument('--dataset', default='shanghai', type=str)
    parser.add_argument('--consecutive', default=4, type=int, help='')
    parser.add_argument('--save_image', default=False, type=str2bool, nargs='?', const=True, help='save frame with bbox for specific frame')
    parser.add_argument('--save_image_all', default=False, type=str2bool, nargs='?', const=True, help='save resized bbox for all image')
    parser.add_argument('--train_od', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--test_od', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--is_save_train_pickle', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--is_save_test_pickle', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--is_load_train_pickle', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--is_load_test_pickle', default=False, type=str2bool, nargs='?', const=True)

    args = parser.parse_args()
    cfg = update_config(args, mode='object_detection')
    cfg.print_cfg() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    # make and set dirs
    makedirs(cfg)
    train_folders, train_length, test_folders, test_length = setdirs(cfg)

    # object detection
    ObjectDetection(cfg, device, train_folders, train_length, test_folders, test_length)


if __name__=="__main__":
    main()
