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


IN_SHAPE = (224, 224, 3)  # shape of input image tensor
NUM_CLASSES = 1000        # number of output classes (1000 for ImageNet)


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
    """get_batch_size

    These default batch_size values were chosen based on available
    GPU RAM (11GB) on GeForce GTX-2080Ti.
    """
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
    """get_iter_size

    These default iter_size values were chosen to make 'effective'
    batch_size to be 256.
    """
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
    return value if value > 0. else 3e-4


def get_final_lr(model_name, value):
    return value if value > 0. else 3e-4


def get_weight_decay(model_name, value):
    return value if value >= 0. else 1e-5


def get_optimizer(model_name, optim_name, initial_lr, epsilon=1e-2):
    """get_optimizer

    Note that learning rate decay is implemented as a callback in
    the model.fit(), so I do not specify 'decay' in the optimizers
    here.
    """
    if optim_name == 'sgd':
        return tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9)
    elif optim_name == 'adam':
        return tf.keras.optimizers.Adam(lr=initial_lr, epsilon=epsilon)
    else:
        # implementation of 'AdamW' is removed temporarily
        raise ValueError


def get_training_model(model_name, optimizer, iter_size, weight_decay):
    """Build the model to be trained."""
    if model_name.endswith('.h5'):
        # load a saved model
        model = tf.keras.models.load_model(
            model_name,
            custom_objects={'AdamW': AdamW})
    else:
        # initialize the model from scratch
        if model_name == 'mobilenet_v2':
            backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=IN_SHAPE, include_top=False, weights=None)
        elif model_name == 'nasnet_mobile':
            backbone = tf.keras.applications.nasnet.NASNetMobile(
                input_shape=IN_SHAPE, include_top=False, weights=None)
        elif model_name == 'resnet50':
            backbone = tf.keras.applications.resnet50.ResNet50(
                input_shape=IN_SHAPE, include_top=False, weights=None)
        else:
            raise ValueError
        # Add a Dropout layer before the final Dense output
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(
            NUM_CLASSES, activation='softmax', name='Logits')(x)
        model = tf.keras.models.Model(inputs=backbone.input, outputs=x)

    if weight_decay > 0.:
        _set_l2(model, weight_decay)
    if iter_size > 1:
        optimizer = convert_to_accum_optimizer(optimizer, iter_size)

    # make sure all layers are set to be trainable
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    print(model.summary())

    return model
