#!/bin/bash

set -xe

# Train googlenet_bn ILSVRC2012 model in 60 epochs.
# Learning rates: exponential decay from 1e-2 to 1e-6

python3 train.py --optimizer adam \
                 --batch_size 64 --iter_size 1 \
                 --lr_sched exp --initial_lr 1e-2 \
                 --lr_decay 0.8576958985908941 \
                 --weight_decay 1e-4 --epochs 60 \
                 googlenet_bn

cp saves/googlenet_bn-model-final.h5 saves/googlenet_bn-stage1.h5
