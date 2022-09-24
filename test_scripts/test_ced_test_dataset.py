import math
import os
import random
import torchvision.utils

from basicsr.data import build_dataloader, build_dataset


def main(mode='folder'):
    """Test reds dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'val'

    opt['name'] = 'CED11'
    opt['type'] = 'CEDOnlyFramesTestDataset'
    # opt['test_mode'] = False
    opt['dataroot_gt'] = 'datasets/CED_h5/HR'
    opt['dataroot_lq'] = 'datasets/CED_h5/LR'
    opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_CED_h5_test.txt'
    opt['io_backend'] = dict(type='hdf5')

    opt['num_worker_per_gpu'] = 16
    opt['batch_size_per_gpu'] = 4
    opt['scale'] = 2

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    print(len(dataset))
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    # nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    # padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 10:
            break
        # print(i)

        lq = data['lq']
        gt = data['gt']
        folder = data['folder']
        print(lq.shape)
        print(gt.shape)
        print(folder)
        nrow = 4
        padding = 0
        for j in range(16):
            torchvision.utils.save_image(
                lq[:, j, :, :, :], f'tmp/lq_{i:03d}_frame{j}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    random.seed(0)
    main()
