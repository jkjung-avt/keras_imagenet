#!/bin/bash

set -xe

# Train googlenetx ILSVRC2012 model in 120 epochs.

python3 train.py --optimizer adam --batch_size 32 --iter_size 1 --lr_sched linear --initial_lr 1e-2 --final_lr 1e-4 --weight_decay 1e-4 --epochs 120 googlenetx
