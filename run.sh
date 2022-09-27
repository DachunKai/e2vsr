#!/bin/bash
sed -i -e 's/\r$//' ./scripts/dist_train.sh && \
./scripts/dist_train.sh 4 options/train/BasicVSR/train_BasicVSR_CED.yml