from glob import glob
import os
from extra.utils import dict2class

share_config = {'mode': 'training',
                'dataset': 'shanghai',
                'coi' : [0, 1, 2, 3, 7],  # person, bicycle, car, motorcycle, truck
                'work_dir': 'anonymous/',
                'data_root': 'anonymous/'}  # remember the final '/'

def update_config(args=None, mode=None):
    # make working directory
    if args.work_num != -1:
        share_config['work_dir'] = f"{share_config['work_dir']}{args.work_num}/"
        if not os.path.exists(share_config['work_dir']):
            os.makedirs(share_config['work_dir'])

    share_config['mode'] = mode
    assert args.dataset in ('avenue', 'shanghai', 'iitb'), 'Dataset error.'
    share_config['dataset'] = args.dataset
    share_config['consecutive'] = args.consecutive
    share_config['save_image'] = args.save_image
    share_config['save_image_all'] = args.save_image_all
    share_config['train_od'] = args.train_od
    share_config['test_od'] = args.test_od

    share_config['is_save_train_pickle'] = args.is_save_train_pickle
    share_config['is_save_test_pickle'] = args.is_save_test_pickle
    share_config['is_load_train_pickle'] = args.is_load_train_pickle
    share_config['is_load_test_pickle'] = args.is_load_test_pickle

    if share_config['is_save_train_pickle'] or share_config['is_save_test_pickle'] or \
    share_config['is_load_train_pickle'] or share_config['is_load_test_pickle'] or \
    share_config['save_image'] or share_config['save_image_all']:
        share_config['train_od'] = False
        share_config['test_od'] = False


    if args.dataset == 'shanghai':
        share_config['train_data'] = share_config['data_root'] + share_config['dataset']+ '/training/'
        share_config['test_data'] = share_config['data_root'] + share_config['dataset']+ '/testing/'
        share_config['max_w'] = 856
        share_config['max_h'] = 480
        share_config['factor_x'] = 1.2
        share_config['factor_y'] = 1.2
        share_config['confidence'] = 0.8
        share_config['test_confidence'] = 0.6
        share_config['obj_size'] = (224, 224)

    elif args.dataset == 'iitb':
        share_config['train_data'] = share_config['data_root'] + share_config['dataset']+ '/training/'
        share_config['test_data'] = share_config['data_root'] + share_config['dataset']+ '/testing/'
        share_config['max_w'] = 1920
        share_config['max_h'] = 1080
        share_config['factor_x'] = 1.0
        share_config['factor_y'] = 1.0
        share_config['confidence'] = 0.8
        share_config['test_confidence'] = 0.6
        share_config['obj_size'] = (224, 224)

    else:
        share_config['train_data'] = share_config['data_root'] + share_config['dataset']+ '/training/frames/'
        share_config['test_data'] = share_config['data_root'] + share_config['dataset']+ '/testing/frames/'
        share_config['factor_x'] = 1.0
        share_config['factor_y'] = 1.0
        share_config['confidence'] = 0.7
        share_config['test_confidence'] = 0.6
        share_config['obj_size'] = (64, 64)
        share_config['max_w'] = 640
        share_config['max_h'] = 360

    return dict2class(share_config)
