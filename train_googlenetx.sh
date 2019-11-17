#!/bin/bash

set -xe

# Train googlenetx ILSVRC2012 model.

python3 train.py --optimizer adam --batch_size 64 --iter_size 1 --lr_sched exp --initial_lr 1e-2 --lr_decay 0.8413951416451951 --weight_decay 1e-4 --epochs 40 googlenetx
