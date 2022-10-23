import math
import os
import random

from basicsr.data import build_dataloader, build_dataset


def main(mode='folder'):
    """Test reds dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'val'

    opt['name'] = 'Vid4'
    opt['type'] = 'Vid4WithEventsTestDataset'
    # opt['test_mode'] = False
    opt['dataroot_gt'] = 'datasets/Vid4_h5/Voxel_3/HR'
    opt['dataroot_lq'] = 'datasets/Vid4_h5/Voxel_3/LRx4'
    opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_Vid4_h5.txt'
    opt['io_backend'] = dict(type='hdf5')

    opt['num_worker_per_gpu'] = 2
    opt['batch_size_per_gpu'] = 4
    opt['scale'] = 4
    opt['is_event'] = True

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    print(len(dataset))
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    # nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    # padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        # if i > 10:
        #     break
        # print(i)

        lq = data['lq']
        gt = data['gt']
        event_lq = data['event_lq']
        folder = data['folder']
        print("lq.shape: ", lq.shape)
        print("gt.shape: ", gt.shape)
        print("event_lq.shape: ", event_lq.shape)
        print("folder: ", folder)
        nrow = 4
        padding = 0
        # assert lq.shape[1] == event_lq.shape[1] + 1, "shape error"
        # for j in range(16):
        #     torchvision.utils.save_image(
        #         lq[:, j, :, :, :], f'tmp/lq_{i:03d}_frame{j}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    random.seed(0)
    main()
