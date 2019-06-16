"""models.py

Implemented models:
    1. mobilenet_v2
    2. ...... (more to be added)
"""


import tensorflow as tf

from config import config
from utils.optimizer import convert_to_accum_optimizer


# Constants (TO-DO: different settings for different models?)
INITIAL_LR = 4e-4    # Initial learning rate
LR_DECAY = 1e-6      # Learning rate decay


def get_mobilenet_v2():
    """Build mobilenet_v2 training model."""
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=True,
        weights=None,
        classes=1000)
    # TO-DO: add weight decay here
    optimizer = convert_to_accum_optimizer(
        tf.keras.optimizers.Adam(lr=INITIAL_LR, decay=LR_DECAY),
        config.ITER_SIZE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def get_training_model(model_name):
    """Build the model to be trained."""
    if model_name == 'mobilenet_v2':
        model = get_mobilenet_v2()
    else:
        raise ValueError

    print(model.summary())
    return model
