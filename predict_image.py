"""predict_image.py

This script is for testing a trained Keras ImageNet model.

Example usage:
   $ python3 predict_image.py saves/googlenet_bn-model-final.h5 \
                              sample.jpg
"""


import argparse

import numpy as np
import cv2
import tensorflow as tf

from utils.utils import config_keras_backend, clear_keras_session


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help='e.g. saves/googlenet_bn-model-final.h5')
    parser.add_argument('jpg',
                        help='an image file to be predicted')
    args = parser.parse_args()
    return args


def preprocess(img):
    """Preprocess an image for Keras ImageNet model inferencing."""
    if img.ndim != 3:
        raise TypeError('bad ndim of img')
    if img.dtype != np.uint8:
        raise TypeError('bad dtype of img')
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img *= (2.0/255)  # normalize to: 0.0~2.0
    img -= 1.0        # subtract mean to make it: -1.0~1.0
    img = np.expand_dims(img, axis=0)
    return img


def main():
    args = parse_args()

    # load the cls_list (index to class name)
    with open('data/synset_words.txt') as f:
        cls_list = sorted(f.read().splitlines())

    config_keras_backend()

    # load the trained model
    net = tf.keras.models.load_model(args.model)

    # load and preprocess the test image
    img = cv2.imread(args.jpg)
    if img is None:
        raise SystemExit('cannot load the test image: %s' % args.jpg)
    img = preprocess(img)

    # predict and postprocess
    pred = net.predict(img)[0]
    top5_idx = pred.argsort()[::-1][:5]  # take the top 5 predictions
    for i in top5_idx:
        print('%5.2f   %s' % (pred[i], cls_list[i]))

    clear_keras_session()


if __name__ == '__main__':
    main()
