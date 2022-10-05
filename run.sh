# !/bin/bash
python -W ignore scripts/data_preparation/create_hdf5.py &&
CUDA_VISIBLE_DEVICES=0,2 ./scripts/dist_train.sh 2 options/train/BasicVSR/train_e2vsr_CED.yml