#!/bin/bash

set -xe

# Train resnet50 ILSVRC2012 model in 80 epochs.
# Learning rates: starting from 1e-2, exponentially decaying to 1e-5

python3 train.py --optimizer adam --use_lookahead --batch_size 16 --iter_size 1 --lr_sched exp --initial_lr 1e-2 --lr_decay 0.9172759353897796 --weight_decay 1e-4 --epochs 80 resnet50

cp saves/resnet50-model-final.h5 saves/resnet50-stage1.h5
