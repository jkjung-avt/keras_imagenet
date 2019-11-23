#!/bin/bash

set -xe

# Train googlenet_bn ILSVRC2012 model.

python3 train.py --optimizer adam --use_lookahead --batch_size 64 --iter_size 1 --lr_sched exp --initial_lr 1e-2 --lr_decay 0.7943282347242815 --weight_decay 2.5e-4 --epochs 20 googlenet_bn
