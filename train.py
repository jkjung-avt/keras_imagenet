"""train.sh

This script is used to train the ImageNet models.

Example usage:
  $ python3 train.sh mobilenet_v2
"""


import os
import time
import argparse

import tensorflow as tf

from config import config
from utils.image_processing import preprocess_image
from models.models import get_batch_size, get_iter_size
from models.models import get_initial_lr, get_final_lr
from models.models import get_weight_decay, get_training_model


def config_keras_backend():
    """Config tensorflow backend to use less GPU memory."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height
        # and width that is set dynamically by decode_jpeg. In other
        # words, the height and width of image is unknown at compile-i
        # time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).
        # The various adjust_* ops all require this range for dtype
        # float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def resize_and_rescale_image(image, height, width, scope=None):
    """Prepare one image for training/evaluation.

    Args:
        image: 3-D float Tensor
        height: integer
        width: integer
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor of prepared image.
    """
    with tf.name_scope(values=[image, height, width], name=scope,
                       default_name='resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        # rescale to [-1,1]
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
    return image


def _parse_fn(example_serialized, is_training):
    """Helper function for parse_fn_train() and parse_fn_valid()

    Each Example proto (TFRecord) contains the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

    Args:
        example_serialized: scalar Tensor tf.string containing a
        serialized Example protocol buffer.

    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    parsed = tf.parse_single_example(example_serialized, feature_map)
    image = decode_jpeg(parsed['image/encoded'])
    if config.DATA_AUGMENTATION:
        image = preprocess_image(image, 224, 224, is_training=is_training)
    else:
        image = resize_and_rescale_image(image, 224, 224)
    label = tf.one_hot(parsed['image/class/label'], 1000, dtype=tf.float32)
    return (image, label)


def parse_fn_train(example_serialized):
    """Parses an Example proto containing a training image."""
    return _parse_fn(example_serialized, is_training=True)


def parse_fn_valid(example_serialized):
    """Parses an Example proto containing a validation image."""
    return _parse_fn(example_serialized, is_training=False)


def get_dataset(tfrecords_dir, subset, batch_size):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    files = tf.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=8192)
    parser = parse_fn_train if subset == 'train' else parse_fn_valid
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=parser,
            batch_size=batch_size,
            num_parallel_calls=config.NUM_DATA_WORKERS))
    dataset = dataset.prefetch(batch_size)
    return dataset


def get_lrate_func(initial_lr, final_lr, total_epochs):
    def step_decay(epoch):
        """Decay LR linearly for each epoch."""
        ratio = max((total_epochs - 1 - epoch) / (total_epochs - 1), 0.)
        lr = final_lr + (initial_lr - final_lr) * ratio
        print('Epoch %d, lr = %f' % (epoch, lr))
        return lr
    return step_decay


def train(model_name, batch_size, iter_size, initial_lr, final_lr,
          weight_decay, epochs):
    """Prepare data and train the model."""
    batch_size = get_batch_size(model_name, batch_size)
    iter_size = get_iter_size(model_name, iter_size)
    initial_lr = get_initial_lr(model_name, initial_lr)
    final_lr = get_final_lr(model_name, final_lr)
    weight_decay = get_weight_decay(model_name, weight_decay)

    # get trainig and validation data
    ds_train = get_dataset(
        '/ssd/jkjung/data/ILSVRC2012/tfrecords', 'train', batch_size)
    ds_validation = get_dataset(
        '/ssd/jkjung/data/ILSVRC2012/tfrecords', 'validation', batch_size)

    # instiante training callbacks
    lrate = tf.keras.callbacks.LearningRateScheduler(
        get_lrate_func(initial_lr, final_lr, epochs))
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        '{}/{}'.format(config.SAVE_DIR, model_name) + '-ckpt-{epoch:03d}.h5',
        monitor='val_loss',
        save_best_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='{}/{}'.format(config.LOG_DIR, time.time()))

    # build model and do training
    model = get_training_model(
        model_name, iter_size, initial_lr, weight_decay)
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
    model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, model_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--iter_size', type=int, default=-1)
    parser.add_argument('--initial_lr', type=float, default=-1.)
    parser.add_argument('--final_lr', type=float, default=-1.)
    parser.add_argument('--weight_decay', type=float, default=-1.)
    parser.add_argument('--epochs', type=int, default=1,
                        help='total number of epochs for training [1]')
    parser.add_argument('model', type=str,
                        help='mobilenet_v2, nasnet_mobile or resnet50')
    args = parser.parse_args()
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    config_keras_backend()
    train(args.model, args.batch_size, args.iter_size,
          args.initial_lr, args.final_lr, args.weight_decay,
          args.epochs)


if __name__ == '__main__':
    main()
