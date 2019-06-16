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
from utils.optimizer import convert_to_accum_optimizer


# Constants



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


def resize_image(image, height, width, scope=None):
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
        # Crop the central region of the image with an area containing
        # 87.5% of the original image.
        #image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def parse_fn(example_serialized):
    """Parses an Example proto containing a training example of an image.

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
    image = resize_image(image, 224, 224)
    # rescale to [-1,1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    label = tf.one_hot(parsed['image/class/label'], 1000, dtype=tf.float32)
    return (image, label)


def get_dataset(tfrecords_dir, subset):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    files = tf.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=8192)
    #dataset = dataset.apply(
    #    tf.data.experimental.shuffle_and_repeat(
    #        buffer_size=8192))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=parse_fn,
            batch_size=config.BATCH_SIZE,
            num_parallel_calls=config.NUM_DATA_WORKERS))
    dataset = dataset.prefetch(config.BATCH_SIZE)
    return dataset


def get_training_model(model_name):
    """Build the model to be trained."""
    if model_name == 'mobilenet_v2':
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=True,
            weights=None,
            classes=1000)
        optimizer = convert_to_accum_optimizer(
            tf.keras.optimizers.Adam(lr=config.INITIAL_LR, decay=config.LR_DECAY),
            config.ITER_SIZE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        raise ValueError

    print(model.summary())
    return model


def train(model_name):
    """Prepare data and train the model."""
    ds_train = get_dataset(
        '/ssd/jkjung/data/ILSVRC2012/tfrecords', 'train')
    ds_validation = get_dataset(
        '/ssd/jkjung/data/ILSVRC2012/tfrecords', 'validation')

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        '{}/{}'.format(config.SAVE_DIR, model_name) + '-ckpt-{epoch:03d}.h5',
        monitor='val_loss',
        save_best_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='{}/{}'.format(config.LOG_DIR, time.time()))

    model = get_training_model(model_name)
    model.fit(
        x=ds_train,
        steps_per_epoch=1281167 // config.BATCH_SIZE,
        validation_data=ds_validation,
        validation_steps=50000 // config.BATCH_SIZE,
        callbacks=[model_checkpoint, tensorboard],
        # The following doesn't seem to help.
        # use_multiprocessing=True, workers=4,
        epochs=config.EPOCHS)

    model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, model_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    args = parser.parse_args()
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    config_keras_backend()
    train(args.model)


if __name__ == '__main__':
    main()
