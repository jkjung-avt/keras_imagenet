"""models.py

Implemented models:
    1. MobileNetV2
    2. NASNetMobile
    3. ResNet50
"""


import tensorflow as tf

from config import config
from utils.optimizer import convert_to_accum_optimizer


def _set_l2(model, weight_decay):
    """Add L2 regularization into layers with weights

    Reference: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.gamma))


def _get_model(model_func, initial_lr, iter_size, weight_decay):
    """Build keras model."""
    model = model_func(include_top=True, weights=None, classes=1000)
    if (weight_decay > 0):
        _set_l2(model, weight_decay)

    # TO-DO: add weight decay here
    amsgrad = config.ADAM_USE_AMSGRAD
    optimizer = convert_to_accum_optimizer(
        tf.keras.optimizers.Adam(lr=initial_lr, amsgrad=amsgrad),
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


def get_final_lr(model_name, value):
    return value if value > 0 else 1e-5


def get_weight_decay(model_name, value):
    return value if value > 0 else \
           (1e-6 if 'mobilenet' in model_name else 1e-5)


def get_training_model(model_name, iter_size, initial_lr, weight_decay):
    """Build the model to be trained."""
    if model_name == 'mobilenet_v2':
        model = _get_model(tf.keras.applications.mobilenet_v2.MobileNetV2,
                           initial_lr, iter_size, weight_decay)
    elif model_name == 'nasnet_mobile':
        model = _get_model(tf.keras.applications.nasnet.NASNetMobile,
                           initial_lr, iter_size, weight_decay)
    elif model_name == 'resnet50':
        model = _get_model(tf.keras.applications.resnet50.ResNet50,
                           initial_lr, iter_size, weight_decay)
    else:
        raise ValueError
    print(model.summary())
    return model
