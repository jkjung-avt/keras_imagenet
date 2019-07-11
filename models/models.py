"""models.py

Implemented models:
    1. MobileNetV2 ('mobilenet_v2')
    2. NASNetMobile ('nasnet_mobile')
    3. ResNet50 ('resnet50')
"""


import tensorflow as tf

from config import config
from models.adamw import AdamW
from models.optimizer import convert_to_accum_optimizer


def _set_l2(model, weight_decay):
    """Add L2 regularization into layers with weights

    Reference: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        elif isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.gamma))


def get_batch_size(model_name, value):
    if value > 0:
        return value
    elif 'mobilenet_v2' in model_name:
        return 64
    elif 'nasnet_mobile' in model_name:
        return 32
    elif 'resnet50' in model_name:
        return 16
    else:
        raise ValueError


def get_iter_size(model_name, value):
    if value > 0:
        return value
    elif 'mobilenet_v2' in model_name:
        return 4
    elif 'nasnet_mobile' in model_name:
        return 8
    elif 'resnet50' in model_name:
        return 16
    else:
        raise ValueError


def get_initial_lr(model_name, value):
    return value if value > 0 else 3e-4


def get_final_lr(model_name, value):
    return value if value > 0 else 1e-5


def get_weight_decay(model_name, value):
    return value if value > 0 else 1e-5


def get_training_model(model_name, iter_size, initial_lr, weight_decay,
                       use_weight_decay=True, use_l2_regularization=False,
                       weight_decay_normalizer=1.):
    if use_weight_decay + use_l2_regularization >= 2:
        raise ValueError
    """Build the model to be trained."""
    if model_name.endswith('.h5'):
        model = tf.keras.models.load_model(model_name)
    elif model_name == 'mobilenet_v2':
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=True, weights=None, classes=1000)
    elif model_name == 'nasnet_mobile':
        model = tf.keras.applications.nasnet.NASNetMobile(
            include_top=True, weights=None, classes=1000)
    elif model_name == 'resnet50':
        model = tf.keras.applications.resnet50.ResNet50(
            include_top=True, weights=None, classes=1000)
    else:
        raise ValueError

    if use_weight_decay:
        optimizer = AdamW(lr=initial_lr,
                          weight_decay=weight_decay,
                          weight_decay_normalizer=weight_decay_normalizer)
    else:
        if use_l2_regularization:
            if (weight_decay > 0):
                _set_l2(model, weight_decay)
        optimizer = tf.keras.optimizers.Adam(lr=initial_lr)
    if iter_size > 1:
        optimizer = convert_to_accum_optimizer(optimizer, iter_size)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
