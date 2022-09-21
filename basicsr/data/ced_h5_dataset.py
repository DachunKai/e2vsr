import os
import numpy as np
import os.path as osp
import random
import torch
import math
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, img2tensor
from basicsr.data import build_dataloader, build_dataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CEDOnlyFramesDataset(data.Dataset):
    """CED dataset for training recurrent networks

    keys is generated from CED_h5_train.txt, example:
    keys[0] = 'simple_flowers_infrared.h5/000000'

    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.num_frame = opt['num_frame']

        self.keys = []
        self.frame_num_dict = {}

        # train/test will have different meta_info_file
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                name, num = line.split(' ')
                self.keys.extend([f'{name}/{i:06d}' for i in range(int(num))])
                self.frame_num_dict[name] = num.strip()
        fin.close()

        ## So here, we can get self.keys(list), self.frame_num(dict)
        ## Example:
        ## self.keys[0] = 'simple_flowers_infrared.h5/000000
        ## self.frame_num['simple_flowers_infrared.h5'] = '1039'

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
        clip_name, frame_name = osp.dirname(key), osp.basename(key)
        self.io_backend_opt['h5_clip'] = clip_name
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # determine the neighboring frames

        interval = random.choice(self.interval_list)
        start_frame_idx = int(frame_name)
        clip_num = int(self.frame_num_dict[clip_name])
        if start_frame_idx > clip_num - self.num_frame * interval:
            start_frame_idx = random.randint(0, clip_num - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        img_lqs, img_gts = self.file_client.get(neighbor_list)

        # randomly crop
        img_lqs, img_gts = paired_random_crop(img_gts, img_lqs, gt_size, scale)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': key}


# @DATASET_REGISTRY.register()
# class CEDWithEventsDataset(data.Dataset):
#     raise NotImplementedError

if __name__ == '__main__':
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'CED'
    opt['type'] = 'CEDOnlyFramesDataset'
    opt['test_mode'] = False
    opt['dataroot_gt'] = 'datasets/CED_h5/HR'
    opt['dataroot_lq'] = 'datasets/CED_h5/LR'
    opt['meta_info_file'] = 'basicsr/data/meta_info/CED_h5_train.txt'
    opt['io_backend'] = dict(type='hdf5')

    opt['num_frame'] = 5
    opt['gt_size'] = 128
    opt['interval_list'] = [1]
    opt['random_reverse'] = True
    opt['use_hflip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 2

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        key = data['key']
        print(lq.shape)
        # for j in range(opt['num_frame']):
        #     torchvision.utils.save_image(
        #         lq[:, j, :, :, :], f'tmp/lq_{i:03d}_frame{j}.png', nrow=nrow, padding=padding, normalize=False)
        #     torchvision.utils.save_image(
        #         gt[:, j, :, :, :], f'tmp/gt_{i:03d}_frame{j}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)