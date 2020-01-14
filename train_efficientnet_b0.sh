#!/bin/bash

set -xe

# Train efficientnet_b0 ILSVRC2012 model in 60 epochs.
# Learning rates: exponential decay from 1e-2 to 3e-5

python3 train.py --dropout_rate 0.4 --optimizer adam \
                 --batch_size 64 --iter_size 1 \
                 --lr_sched exp --initial_lr 1e-2 \
                 --lr_decay 0.9077202433980235 \
                 --weight_decay 1e-4 --epochs 60 \
                 efficientnet_b0

cp saves/efficientnet_b0-model-final.h5 saves/efficientnet_b0-stage1.h5
