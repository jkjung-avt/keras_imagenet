#!/bin/bash

set -xe

# Train googlenet_bn ILSVRC2012 model.
# Total epochs: 20 + 40 = 60

python3 train.py --optimizer adam --use_lookahead --batch_size 64 --iter_size 1 --lr_sched exp --initial_lr 1e-2 --lr_decay 0.7943282347242815 --weight_decay 2.5e-4 --epochs 20 googlenet_bn

mv saves/googlenet_bn-model-final.h5 saves/googlenet_bn-stage1.h5
rm saves/googlenet_bk-ckpt-*

python3 train.py --optimizer sgd --use_lookahead --batch_size 64 --iter_size 1 --lr_sched linear --initial_lr 1e-4 --final_lr 1e-6 --weight_decay 2.5e-4 --epochs 40 saves/googlenet_bn-stage1.h5

cp saves/googlenet_bn-model-final.h5 saves/googlenet_bn-stage2.h5
