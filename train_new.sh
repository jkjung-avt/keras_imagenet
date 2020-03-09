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
        # Learning rates: exponential decay from 1e-2 to 1e-4
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9261187281287935 \
                         --weight_decay 2e-4 --epochs 60 mobilenet_v2
        ;;
    resnet50 )
        # Learning rates: exponential decay from 1e-2 to 1e-5
        python3 train.py --dropout_rate 0.5 \
                         --optimizer adam --batch_size 16 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.8912509381337456 \
                         --weight_decay 2e-4 --epochs 80 resnet50
        ;;
    googlenet_bn )
        # Learning rates: exponential decay from 1e-2 to 1e-5
        python3 train.py --dropout_rate 0.4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.8912509381337456 \
                         --weight_decay 2e-4 --epochs 60 googlenet_bn
        ;;
    inception_v2 )
        # Learning rates: exponential decay from 1e-2 to 1e-5
        python3 train.py --dropout_rate 0.4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.8912509381337456 \
                         --weight_decay 2e-4 --epochs 60 inception_v2
        ;;
    efficientnet_b0 )
        # Learning rates: exponential decay from 1e-2 to 3e-5
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9077202433980235 \
                         --weight_decay 1e-4 --epochs 60 \
                         efficientnet_b0
        ;;
    efficientnet_b1 )
        # Learning rates: exponential decay from 1e-2 to 3e-5
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9077202433980235 \
                         --weight_decay 1e-4 --epochs 60 \
                         efficientnet_b1
        ;;
    efficientnet_b4 )
        # Learning rates: exponential decay from 1e-2 to 3e-5
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 \
                         --lr_decay 0.9077202433980235 \
                         --weight_decay 1e-4 --epochs 60 \
                         efficientnet_b4
        ;;
    * )
        usage
        exit
esac
