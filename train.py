"""train.sh

This script is used to train the ImageNet models.
"""


import os
import time
import argparse

import tensorflow as tf

from config import config
from utils.dataset import get_dataset
from models.models import get_batch_size, get_iter_size
from models.models import get_initial_lr, get_final_lr
from models.models import get_weight_decay, get_training_model


DESCRIPTION = """For example:
$ python3 train.py --dataset_dir  ${HOME}/data/ILSVRC2012/tfrecords \
                   --batch_size   64 \
                   --iter_size    4 \
                   --initial_lr   3e-4 \
                   --final_lr     1e-5 \
                   --weight_decay 1e-5 \
                   --epochs       50 \
                   mobilenet_v2
"""
SUPPORTED_MODELS = (
    '"mobilenet_v2", "nasnet_mobile", "resnet50" or just specify '
    'a saved Keras model (.h5) file')


def config_keras_backend():
    """Config tensorflow backend to use less GPU memory."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)


def get_lrate_func(initial_lr, final_lr, total_epochs):
    def step_decay(epoch):
        """Decay LR linearly for each epoch."""
        ratio = max((total_epochs - 1 - epoch) / (total_epochs - 1), 0.)
        lr = final_lr + (initial_lr - final_lr) * ratio
        # Keras counts the 1st epoch as epoch 1 (not 0)
        print('Epoch %d, lr = %f' % (epoch+1, lr))
        return lr
    return step_decay


def train(model_name, batch_size, iter_size, initial_lr, final_lr,
          weight_decay, epochs, dataset_dir):
    """Prepare data and train the model."""
    batch_size = get_batch_size(model_name, batch_size)
    iter_size = get_iter_size(model_name, iter_size)
    initial_lr = get_initial_lr(model_name, initial_lr)
    final_lr = get_final_lr(model_name, final_lr)
    weight_decay = get_weight_decay(model_name, weight_decay)

    # get training and validation data
    ds_train = get_dataset(dataset_dir, 'train', batch_size)
    ds_validation = get_dataset(dataset_dir, 'validation', batch_size)

    # instantiate training callbacks
    lrate = tf.keras.callbacks.LearningRateScheduler(
        get_lrate_func(initial_lr, final_lr, epochs))
    save_name = model_name if not model_name.endswith('.h5') else \
                os.path.split(model_name)[-1].split('.')[0].split('-')[0]
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        '{}/{}'.format(config.SAVE_DIR, save_name) + '-ckpt-{epoch:03d}.h5',
        monitor='val_loss',
        save_best_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='{}/{}'.format(config.LOG_DIR, time.time()))

    # build model and do training
    weight_decay_normalizer = 1. / (((1281167*epochs) / batch_size) ** .5)
    model = get_training_model(
        model_name=model_name,
        iter_size=iter_size,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        weight_decay_normalizer=weight_decay_normalizer)
    model.fit(
        x=ds_train,
        steps_per_epoch=1281167 // batch_size,
        validation_data=ds_validation,
        validation_steps=50000 // batch_size,
        callbacks=[lrate, model_ckpt, tensorboard],
        # The following doesn't seem to help.
        # use_multiprocessing=True, workers=4,
        epochs=epochs)

    # training finished
    model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, save_name))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset_dir', type=str,
                        default=config.DEFAULT_DATASET_DIR)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--iter_size', type=int, default=-1)
    parser.add_argument('--initial_lr', type=float, default=-1.)
    parser.add_argument('--final_lr', type=float, default=-1.)
    parser.add_argument('--weight_decay', type=float, default=-1.)
    parser.add_argument('--epochs', type=int, default=1,
                        help='total number of epochs for training [1]')
    parser.add_argument('model', type=str,
                        help=SUPPORTED_MODELS)
    args = parser.parse_args()
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    config_keras_backend()
    train(args.model, args.batch_size, args.iter_size,
          args.initial_lr, args.final_lr, args.weight_decay,
          args.epochs, args.dataset_dir)


if __name__ == '__main__':
    main()
