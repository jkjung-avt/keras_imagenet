#!/bin/bash

set -xe

usage()
{
    echo
    echo "Usage: ./trash_new.sh <model_name>"
    echo
    echo "where <model_name> could be one of the following:"
    echo "    mobilenet_v2, resnet50, googlenet_bn, inception_v2,"
    echo "    efficientnet_b0, efficientnet_b1, efficientnet_b4"
    echo
}

if [ $# -ne 1 ]; then
    usage
    exit
fi

case $1 in
    mobilenet_v2 )
        # Learning rates exp decay from 1e-2 to 1e-4, 60 epochs
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9249147277217333 \
                         --weight_decay 4e-5 --epochs 60 mobilenet_v2
        ;;
    resnet50 )
        # Learning rates exp decay from 1e-2 to 1e-5, 80 epochs
        python3 train.py --dropout_rate 0.5 \
                         --optimizer adam --batch_size 16 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9162739011886736 \
                         --weight_decay 2e-4 --epochs 80 resnet50
        ;;
    googlenet_bn )
        # Learning rates exp decay from 1e-2 to 1e-5, 60 epochs
        python3 train.py --dropout_rate 0.4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.8895134973108234 \
                         --weight_decay 2e-4 --epochs 60 googlenet_bn
        ;;
    inception_v2 )
        # Learning rates exp decay from 1e-2 to 1e-5, 60 epochs
        python3 train.py --dropout_rate 0.4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.8895134973108234 \
                         --weight_decay 2e-4 --epochs 60 inception_v2
        ;;
    efficientnet_b0 )
        # Learning rates exp decay from 1e-2 to 1e-4, 60 epochs
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9249147277217333 \
                         --weight_decay 1e-4 --epochs 60 \
                         efficientnet_b0
        ;;
    efficientnet_b1 )
        # Learning rates exp decay from 1e-2 to 1e-4, 60 epochs
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9249147277217333 \
                         --weight_decay 1e-4 --epochs 60 \
                         efficientnet_b1
        ;;
    efficientnet_b4 )
        # Learning rates exp decay from 1e-2 to 1e-4, 80 epochs
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 16 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9433732216299777 \
                         --weight_decay 1e-4 --epochs 80 \
                         efficientnet_b4
        ;;
    * )
        usage
        exit
esac
