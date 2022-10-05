import os
import os.path as osp
from glob import glob
import time
from multiprocessing import Pool


#input list is [''datasets/CED/HR/CED_simple/simple_rabbits', 'datasets/CED_2/LRx4/CED_simple/simple_rabbits']
def process_dir(list):
    print(list[0])

    events_folder = osp.join(list[0], 'events')
    images_folder = osp.join(list[1], 'images')
    empty_list = []
    event_list = sorted(glob(osp.join(events_folder, '*.txt')))
    for idx in range(len(event_list)):
        if not osp.getsize(event_list[idx]):
            empty_list.append(idx)
            # os.remove(event_list[idx])
            # print(f"remove empty file {event_list[idx]}")
            os.remove(osp.join(images_folder, f'{(idx+1):06d}.png'))
            print(f"remove empty file {event_list[idx]}")

    with open(osp.join(list[1], 'timestamp.txt'), 'r') as f:
        lines = f.readlines()
        for i in reversed(empty_list):
            del lines[i+1]
        f = open(osp.join(list[1], 'timestamp.txt'), 'w')
        f.writelines(lines)
        f.close()

    img_list = sorted(glob(osp.join(images_folder, '*.png')))
    for i in range(len(img_list)):
        old_name = img_list[i]
        new_name = osp.join(images_folder, f'{i:06d}.png')
        os.rename(old_name, new_name)

    # event_list = sorted(glob(osp.join(events_folder, '*.txt')))
    # for i in range(len(event_list)):
    #     old_name = event_list[i]
    #     new_name = osp.join(events_folder, f'{i:06d}.txt')
    #     os.rename(old_name, new_name)

    out_message = list[1].split('/')[-2] + '/' + list[1].split('/')[-1]
    print(f"Process {out_message} finished.")


if __name__ == "__main__":

    root_path = 'datasets/CED_2/LRx2'

    pathlist = []
    for root, subdir, _ in os.walk(root_path):
        if 'images' in subdir:
            former = osp.join('datasets/CED/HR', root.split('/CED_2/LRx2/')[1])
            pathlist.append([former, root])

    T1 = time.time()

    pool = Pool(40)
    pool.map(process_dir, pathlist)
    pool.close()
    pool.join()
