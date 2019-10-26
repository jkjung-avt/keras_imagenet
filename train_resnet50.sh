#!/bin/bash

set -xe

# Train resnet50 ILSVRC2012 model in 3 stages.
# Learning rates for the 3 stages: 1e-3, 1e-4, 1e-5
# Total 20 + 20 + 20 = 60 epochs.

python3 train.py --optimizer adam --initial_lr 1e-3 --final_lr 1e-3 --weight_decay 0.0 --epochs 20 resnet50
cp saves/resnet50-model-final.h5 saves/resnet50-stage1.h5

python3 train.py --optimizer adam --initial_lr 1e-4 --final_lr 1e-4 --weight_decay 0.0 --epochs 20 saves/resnet50-stage1.h5
cp saves/resnet50-model-final.h5 saves/resnet50-stage2.h5

python3 train.py --optimizer adam --initial_lr 1e-5 --final_lr 1e-5 --weight_decay 0.0 --epochs 20 saves/resnet50-stage2.h5
cp saves/resnet50-model-final.h5 saves/resnet50-stage3.h5
