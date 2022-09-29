#!/bin/bash
pip install -r requirements.txt && \
python setup.py develop && \
sed -i -e 's/\r$//' ./scripts/dist_train.sh && \
pip install einops matplotlib && \
./scripts/dist_train.sh 4 options/train/BasicVSR/train_e2vsr_CED.yml