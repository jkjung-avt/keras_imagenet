#!/bin/bash
set -xe

# Train resnet50 ILSVRC2012 model in 4 stages.
# Total 20 + 40 + 60 + 80 = 200 epochs.

python3 train.py --batch_size 16 --iter_size 16 --initial_lr 3e-4 --final_lr 1e-6 --weight_decay 0.0025 --epochs 20 resnet50

cp saves/resnet50-model-final.h5 saves/resnet50-stage1.h5

python3 train.py --batch_size 16 --iter_size 16 --initial_lr 3e-4 --final_lr 1e-6 --weight_decay 0.0025 --epochs 40 saves/resnet50-stage1.h5

cp saves/resnet50-model-final.h5 saves/resnet50-stage2.h5

python3 train.py --batch_size 16 --iter_size 16 --initial_lr 3e-4 --final_lr 1e-6 --weight_decay 0.0025 --epochs 60 saves/resnet50-stage2.h5

cp saves/resnet50-model-final.h5 saves/resnet50-stage3.h5

python3 train.py --batch_size 16 --iter_size 16 --initial_lr 3e-4 --final_lr 1e-6 --weight_decay 0.0025 --epochs 80 saves/resnet50-stage3.h5

cp saves/resnet50-model-final.h5 saves/resnet50-stage4.h5
