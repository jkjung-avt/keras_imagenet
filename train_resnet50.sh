#!/bin/bash

set -xe

python3 train.py --optimizer adam --weight_decay 0.0 --epochs 60 resnet50
