#!/bin/bash

set -xe

# Train resnet50 ILSVRC2012 model in 3 stages.
# Learning rates for the 3 stages: 3e-3, 3e-4, 3e-5
# Total 30 + 30 + 30 = 90 epochs.

python3 train.py --optimizer adam --initial_lr 3e-3 --final_lr 3e-3 --weight_decay 0.0 --epochs 30 resnet50
cp saves/resnet50-model-final.h5 saves/resnet50-stage1.h5

python3 train.py --optimizer adam --initial_lr 3e-4 --final_lr 3e-4 --weight_decay 0.0 --epochs 30 saves/resnet50-stage1.h5
cp saves/resnet50-model-final.h5 saves/resnet50-stage2.h5

python3 train.py --optimizer adam --initial_lr 3e-5 --final_lr 3e-5 --weight_decay 0.0 --epochs 30 saves/resnet50-stage2.h5
cp saves/resnet50-model-final.h5 saves/resnet50-stage3.h5
