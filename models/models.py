"""models.py

Implemented models:
    1. mobilenet_v2
    2. ...... (more to be added)
"""


import tensorflow as tf

from config import config
from utils.optimizer import convert_to_accum_optimizer


def _get_model(model_func, initial_lr, lr_decay, iter_size):
    """Build keras model."""
    model = model_func(include_top=True, weights=None, classes=1000)
    # TO-DO: add weight decay here
    optimizer = convert_to_accum_optimizer(
        tf.keras.optimizers.Adam(lr=initial_lr, decay=lr_decay),
        iter_size)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def get_batch_size(model_name, value):
    if value > 0:
        return value
    elif model_name == 'mobilenet_v2':
        return 64
    elif model_name == 'nasnet_mobile':
        return 32
    elif model_name == 'resnet50':
        return 16
    else:
        raise ValueError


def get_iter_size(model_name, value):
    if value > 0:
        return value
    elif model_name == 'mobilenet_v2':
        return 4
    elif model_name == 'nasnet_mobile':
        return 8
    elif model_name == 'resnet50':
        return 16
    else:
        raise ValueError


def get_initial_lr(model_name, value):
    return value if value > 0 else 3e-4


def get_lr_decay(model_name, value):
    return value if value > 0 else 1e-6


def get_training_model(model_name, iter_size, initial_lr, lr_decay):
    """Build the model to be trained."""
    if model_name == 'mobilenet_v2':
        model = _get_model(tf.keras.applications.mobilenet_v2.MobileNetV2,
                           initial_lr, lr_decay, iter_size)
    elif model_name == 'nasnet_mobile':
        model = _get_model(tf.keras.applications.nasnet.NASNetMobile,
                           initial_lr, lr_decay, iter_size)
    elif model_name == 'resnet50':
        model = _get_model(tf.keras.applications.resnet50.ResNet50,
                           initial_lr, lr_decay, iter_size)
    else:
        raise ValueError
    print(model.summary())
    return model
