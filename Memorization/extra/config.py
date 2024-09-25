from glob import glob
import os
from extra.utils import dict2class


share_config = {'mode': 'training',
                'dataset': 'shanghai',
                'coi' : [0, 1, 2, 3, 7],  # person, bicycle, car, motorcycle, truck
                'img_size': (224, 224),
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
    share_config['cnl_pool'] = args.cnl_pool
    share_config['consecutive'] = args.consecutive
    share_config['pool_for_sm'] = args.pool_for_sm
    share_config['spatial_f_coreset'] = args.spatial_f_coreset
    share_config['temporal_f_coreset'] = args.temporal_f_coreset
    share_config['highlevel_f_coreset'] = args.highlevel_f_coreset
    share_config['eps_coreset'] = args.eps_coreset
    share_config['random_projection'] = args.random_projection
    share_config['save_feature'] = args.save_feature
    share_config['save_memory'] = args.save_memory

    if args.dataset == 'shanghai':
        share_config['train_data'] = share_config['data_root'] + share_config['dataset']+ '/training/'
        share_config['test_data'] = share_config['data_root'] + share_config['dataset']+ '/testing/'
        share_config['obj_size'] = (224, 224)
        share_config['clip_model'] = 'RN101'
        share_config['power_n'] = 2

    elif args.dataset == 'iitb':
        share_config['train_data'] = share_config['data_root'] + share_config['dataset']+ '/training/'
        share_config['test_data'] = share_config['data_root'] + share_config['dataset']+ '/testing/'
        share_config['obj_size'] = (224, 224)
        share_config['clip_model'] = 'RN101'
        share_config['power_n'] = 1

    else:
        share_config['train_data'] = share_config['data_root'] + share_config['dataset']+ '/training/frames/'
        share_config['test_data'] = share_config['data_root'] + share_config['dataset']+ '/testing/frames/'
        share_config['obj_size'] = (64, 64)
        share_config['clip_model'] = 'RN101'
        share_config['power_n'] = 4

    return dict2class(share_config)
