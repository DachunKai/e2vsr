import os
import argparse
import os.path as osp

from basicsr.utils import scandir
from basicsr.utils.hdf5_util import make_hdf5_from_folders, make_lr_hdf5_from_folders

def create_hdf5_for_ced_hr():
    """Create hdf5 files for CED dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # HR
    folder_path = 'datasets/CED/HR'
    root_path_list, h5_path_list = prepare_h5_keys_ced_hr(folder_path)
    make_hdf5_from_folders(root_path_list, h5_path_list)

    # # LR
    # folder_path = 'dataset/CED/LR'
    # h5_path = 'datasets/CED_h5/LR_h5'
    # h5_path_list = prepare_h5_keys_ced(folder_path)

def prepare_h5_keys_ced_hr(folder_path):
    """Prepare h5 path list for CED dataset

    Args:
        folder_path (str): CED Folder path

    Returns:
        list[str]: h5 root path list.
        example:
            'datasets/CED/CED_additional_IR_filter/outdoor_jumping_infrared_2.h5
    """
    root_path_list = []
    h5_path_list = []
    for root, subdir, _ in os.walk(folder_path):
        if 'images' in subdir:
            root_path_list.append(root)
            out = osp.join('datasets/CED_h5/Voxel_3/HR', osp.basename(root) + '.h5')
            h5_path_list.append(out)

    return root_path_list, h5_path_list

def prepare_h5_keys_ced_lr(folder_path):
    """Prepare h5 path list for CED dataset

    Args:
        folder_path (str): CED Folder path

    Returns:
        list[str]: h5 root path list.
        example:
            'datasets/CED/CED_additional_IR_filter/outdoor_jumping_infrared_2.h5
    """
    root_path_list = []
    h5_hr_path_list = []
    h5_lr_path_list = []
    for root, subdir, _ in os.walk(folder_path):
        if 'images' in subdir:
            root_path_list.append(root)
            h5_hr_path_list.append(osp.join('datasets/CED_h5/Voxel_3/HR', osp.basename(root) + '.h5'))
            h5_lr_path_list.append(osp.join('datasets/CED_h5/Voxel_3/LRx4', osp.basename(root) + '.h5'))

    return root_path_list, h5_hr_path_list, h5_lr_path_list

def create_hdf5_for_ced_lr():
    """Create hdf5 files for CED dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # HR
    folder_path = 'datasets/CED/LR'
    root_path_list, h5_hr_path_list, h5_lr_path_list = prepare_h5_keys_ced_lr(folder_path)
    make_lr_hdf5_from_folders(root_path_list, h5_hr_path_list, h5_lr_path_list)

    # # LR
    # folder_path = 'dataset/CED/LR'
    # h5_path = 'datasets/CED_h5/LR_h5'
    # h5_path_list = prepare_h5_keys_ced(folder_path)


if __name__ == '__main__':
    create_hdf5_for_ced_lr()
