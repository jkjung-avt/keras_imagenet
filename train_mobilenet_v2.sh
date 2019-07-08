#!/bin/bash
set -xe

# Train mobilenet_v2 ILSVRC2012 model in 4 stages.
# Total 50 * 4 = 200 epochs.

python3 train.py --batch_size 64 --iter_size 1 --initial_lr 3e-4 --final_lr 1e-6 --weight_decay 4e-5 --epochs 50 mobilenet_v2

cp saves/mobilenet_v2-model-final.h5 saves/mobilenet_v2-stage1.h5

python3 train.py --batch_size 64 --iter_size 1 --initial_lr 1e-4 --final_lr 1e-6 --weight_decay 4e-5 --epochs 50 saves/mobilenet_v2-stage1.h5

cp saves/mobilenet_v2-model-final.h5 saves/mobilenet_v2-stage2.h5

python3 train.py --batch_size 64 --iter_size 1 --initial_lr 3e-5 --final_lr 1e-6 --weight_decay 4e-5 --epochs 50 saves/mobilenet_v2-stage2.h5

cp saves/mobilenet_v2-model-final.h5 saves/mobilenet_v2-stage3.h5

python3 train.py --batch_size 64 --iter_size 1 --initial_lr 1e-5 --final_lr 1e-6 --weight_decay 4e-5 --epochs 50 saves/mobilenet_v2-stage3.h5

cp saves/mobilenet_v2-model-final.h5 saves/mobilenet_v2-stage4.h5
