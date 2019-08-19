#!/bin/bash
set -xe

# Train resnet50 ILSVRC2012 model in 2 stages.
# Total 20 + 100 = 120 epochs.

python3 train.py --batch_size 16 --iter_size 16 --initial_lr 3e-4 --final_lr 1e-6 --weight_decay 0.1 --epochs 20 resnet50

cp saves/resnet50-model-final.h5 saves/resnet50-stage1.h5

python3 train.py --batch_size 16 --iter_size 16 --initial_lr 1e-4 --final_lr 1e-6 --weight_decay 0.1 --epochs 100 saves/resnet50-stage1.h5

cp saves/resnet50-model-final.h5 saves/resnet50-stage2.h5
