#!/bin/bash

set -xe

# Train mobilenet_v2 ILSVRC2012 model in 60 epochs
# Learning rates: exponential decay from 0.045 to 1e-5

python3 train.py --optimizer adam --batch_size 64 --iter_size 1 --lr_sched exp --initial_lr 0.045 --lr_decay 0.8691868050680678 --weight_decay 4e-5 --epochs 60 mobilenet_v2

cp saves/mobilenet_v2-model-final.h5 saves/modelnet_v2-stage1.h5
