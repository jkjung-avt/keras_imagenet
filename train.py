"""train.sh

This script is used to train the ImageNet models.
"""


import os
import time
import argparse

import tensorflow as tf

from config import config
from utils.utils import config_keras_backend, clear_keras_session
from utils.dataset import get_dataset
from models.models import get_batch_size
from models.models import get_iter_size
from models.models import get_lr_func
from models.models import get_initial_lr
from models.models import get_final_lr
from models.models import get_lr_decay
from models.models import get_weight_decay
from models.models import get_optimizer
from models.models import get_training_model


DESCRIPTION = """For example:
$ python3 train.py --dataset_dir  ${HOME}/data/ILSVRC2012/tfrecords \
                   --dropout_rate 0.4 \
                   --optimizer    adam \
                   --batch_size   64 \
                   --iter_size    1 \
                   --lr_sched     exp \
                   --initial_lr   1e-2 \
                   --lr_decay 0.8576958985908941 \
                   --epochs       60 \
                   --weight_decay 1e-4 \
                   googlenet_bn
"""
SUPPORTED_MODELS = (
    '"mobilenet_v2", "resnet50", "googlenet_bn", "efficientnet_b0", '
    '"efficientnet_b1", "efficientnet_b4" or just specify a saved '
    'Keras model (.h5) file')


def train(model_name, dropout_rate, optim_name,
          use_lookahead, batch_size, iter_size,
          lr_sched, initial_lr, final_lr, lr_decay,
          weight_decay, epochs, dataset_dir):
    """Prepare data and train the model."""
    batch_size   = get_batch_size(model_name, batch_size)
    iter_size    = get_iter_size(model_name, iter_size)
    initial_lr   = get_initial_lr(model_name, initial_lr)
    final_lr     = get_final_lr(model_name, final_lr)
    lr_decay     = get_lr_decay(model_name, lr_decay)
    optimizer    = get_optimizer(model_name, optim_name, initial_lr)
    weight_decay = get_weight_decay(model_name, weight_decay)

    # get training and validation data
    ds_train = get_dataset(dataset_dir, 'train', batch_size)
    ds_valid = get_dataset(dataset_dir, 'validation', batch_size)

    # instantiate training callbacks
    lrate = get_lr_func(epochs, lr_sched, initial_lr, final_lr, lr_decay)
    save_name = model_name if not model_name.endswith('.h5') else \
                os.path.split(model_name)[-1].split('.')[0].split('-')[0]
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.SAVE_DIR, save_name) + '-ckpt-{epoch:03d}.h5',
        monitor='val_loss',
        save_best_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='{}/{}'.format(config.LOG_DIR, time.time()))

    # build model and do training
    model = get_training_model(
        model_name=model_name,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        use_lookahead=use_lookahead,
        iter_size=iter_size,
        weight_decay=weight_decay)
    model.fit(
        x=ds_train,
        steps_per_epoch=1281167 // batch_size,
        validation_data=ds_valid,
        validation_steps=50000 // batch_size,
        callbacks=[lrate, model_ckpt, tensorboard],
        # The following doesn't seem to help in terms of speed.
        # use_multiprocessing=True, workers=4,
        epochs=epochs)

    # training finished
    model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, save_name))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset_dir', type=str,
                        default=config.DEFAULT_DATASET_DIR)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--use_lookahead', action='store_true')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--iter_size', type=int, default=-1)
    parser.add_argument('--lr_sched', type=str, default='linear',
                        choices=['linear', 'exp'])
    parser.add_argument('--initial_lr', type=float, default=-1.)
    parser.add_argument('--final_lr', type=float, default=-1.)
    parser.add_argument('--lr_decay', type=float, default=-1.)
    parser.add_argument('--weight_decay', type=float, default=-1.)
    parser.add_argument('--epochs', type=int, default=1,
                        help='total number of epochs for training [1]')
    parser.add_argument('model', type=str,
                        help=SUPPORTED_MODELS)
    args = parser.parse_args()

    if args.use_lookahead and args.iter_size > 1:
        raise ValueError('cannot set both use_lookahead and iter_size')

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    config_keras_backend()
    train(args.model, args.dropout_rate, args.optimizer,
          args.use_lookahead, args.batch_size, args.iter_size,
          args.lr_sched, args.initial_lr, args.final_lr, args.lr_decay,
          args.weight_decay, args.epochs, args.dataset_dir)
    clear_keras_session()


if __name__ == '__main__':
    main()
