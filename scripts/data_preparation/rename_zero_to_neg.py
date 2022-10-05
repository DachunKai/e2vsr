from distutils.log import error
import os
import os.path as osp
from glob import glob
import time
from multiprocessing import Pool

def process_dir(list):
    event_list = sorted(glob(osp.join(list[0], '*.txt')))
    for txt_file in event_list:
        change_0_to_neg(txt_file)

    out_message = list[0].split('/')[-2] + '/' + list[0].split('/')[-1]
    print(f"Process {out_message} finished.")

def change_0_to_neg(txt_file):
    to_write_list = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if line[-1] == '0':
                line[-1] = '-1'
            to_write = line[0] + ' ' + line[1]+ ' ' + line[2] + ' ' + line[-1] + '\n'
            to_write_list.append(to_write)
    f.close()

    f = open(txt_file, 'w')
    f.writelines(to_write_list)
    f.close()

if __name__ == '__main__':
    root_path = 'datasets/CED_3/HR'

    dir_list = []
    for root, subdir, _ in os.walk(root_path):
        if 'events' in subdir:
            dir_list.append([osp.join(root, 'events')])

    T1 = time.time()

    pool = Pool(40)
    pool.map(process_dir, dir_list)
    pool.close()
    pool.join()


