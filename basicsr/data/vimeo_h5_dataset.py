import numpy as np
import os.path as osp
import random

random.seed(0)
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, paired_event_random_crop, augment_with_event
from basicsr.utils import FileClient, get_root_logger, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Vimeo90KOnlyFramesDataset(data.Dataset):
    """Vimeo90k septuplet dataset for training recurrent networks with only frames

    keys is generated from meta_info_vimeo_h5_train.txt, example:
    keys[0] = '00001_0001.h5'

    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']

        self.keys = []
        self.phase = opt['phase']

        # train/test will have different meta_info_file
        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]
        fin.close()
        if self.phase == 'train':
            self.num_frame = 13
        else:
            self.num_frame = 7
        self.neighbor_list = [i for i in range(self.num_frame)]

        ## So here, we can get self.keys(list), self.frame_num(dict)
        ## Example:
        ## self.keys[0] = '00001_0001.h5'

        # file client(io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_hdf5 = False
        if self.io_backend_opt['type'] == 'hdf5':
            self.is_hdf5 = True
            self.io_backend_opt['h5_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['LR', 'HR']
        else:
            raise ValueError(f"We don't realize {self.io_backend_opt['type']} backend")

        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        self.io_backend_opt['h5_clip'] = key
        # if self.file_client is None:
        self.file_client = FileClient(self.io_backend_opt['type'], **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        # print("clip_name: ", clip_name, " neighbor_list: ", neighbor_list)
        img_lqs, img_gts = self.file_client.get(self.neighbor_list)

        if self.phase == 'train':
            # train
            # randomly crop
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale)

            # augmentation - flip, rotate
            img_lqs.extend(img_gts)
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

            img_results = img2tensor(img_results)
            img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
            img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        else:
            # test
            img_lqs = torch.stack(img2tensor(img_lqs), dim=0)
            img_gts = torch.stack(img2tensor(img_gts), dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': key}


@DATASET_REGISTRY.register()
class Vimeo90kWithEventsDataset(data.Dataset):
    """Vimeo90k dataset for training recurrent networks with events

    keys is generated from meta_info_vimeo_h5_train.txt, example:
    keys[0] = '00001_0001.h5'
    return:
        data['lq']: [T, C, H, W]
        data['gt']: [T, C, 4H, 4W]
        data['event_lq']: [T-1, C, H, W]
        data['key']: str means clip hdf5 name
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']

        self.keys = []
        self.phase = opt['phase']
        self.is_event = opt['is_event']

        # train/test will have different meta_info_file
        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]
        fin.close()
        if self.phase == 'train':
            self.num_frame = 13
        else:
            self.num_frame = 7
        self.neighbor_list = [i for i in range(self.num_frame)]

        ## So here, we can get self.keys(list)
        ## Example:
        ## self.keys[0] = '00001_0001.h5'

        # file client(io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_hdf5 = False
        if self.io_backend_opt['type'] == 'hdf5':
            self.is_hdf5 = True
            self.io_backend_opt['h5_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['LR', 'HR']
            self.io_backend_opt['is_event'] = self.is_event
        else:
            raise ValueError(f"We don't realize {self.io_backend_opt['type']} backend")

        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        self.io_backend_opt['h5_clip'] = key
        # if self.file_client is None:
        self.file_client = FileClient(self.io_backend_opt['type'], **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        img_lqs, img_gts, event_lqs = self.file_client.get(self.neighbor_list)

        if self.phase == 'train':
            # train
            # randomly crop
            img_gts, img_lqs, event_lqs = paired_event_random_crop(img_gts, img_lqs, event_lqs, gt_size, scale)

            # augmentation - flip, rotate
            img_lqs.extend(img_gts)
            img_results, event_lqs = augment_with_event(img_lqs, event_lqs, self.opt['use_hflip'], self.opt['use_rot'])

            img_results = img2tensor(img_results)
            img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
            img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)
            for i in range(len(event_lqs)):
                if event_lqs[i].shape != (3, 64, 64):
                    print(f"event_lqs[{i}] shape error with shape is: {event_lqs[i].shape}")
                    print(f"clip_name: {key}, neighbor_list: {self.neighbor_list}")
            event_lqs = torch.from_numpy(np.stack(event_lqs, axis=0))

        else:
            # test
            img_lqs = torch.stack(img2tensor(img_lqs), dim=0)
            img_gts = torch.stack(img2tensor(img_gts), dim=0)
            event_lqs = torch.from_numpy(np.stack(event_lqs, axis=0))

        return {'lq': img_lqs, 'gt': img_gts, 'event_lq': event_lqs, 'key': key}