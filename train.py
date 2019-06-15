"""train.sh

This script is used to train the ImageNet models.

Example usage:
  $ python3 train.sh mobilenet_v2
"""


import os
import sys
import time
import tensorflow as tf


NUM_DATA_WORKERS = 4
BATCH_SIZE = 16
EPOCHS = 1


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
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=8192)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=parse_fn,
            batch_size=BATCH_SIZE,
            num_parallel_calls=NUM_DATA_WORKERS))
    dataset = dataset.prefetch(BATCH_SIZE)
    return dataset


def main():
    config_keras_backend()

    ds_train = get_dataset(
        '/ssd/jkjung/data/ILSVRC2012/tfrecords', 'train')
    ds_validation = get_dataset(
        '/ssd/jkjung/data/ILSVRC2012/tfrecords', 'validation')

    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=True, weights=None, classes=1000)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),
        loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'chkpt-{epoch:03d}.h5', monitor='val_loss', save_best_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/{}'.format(time.time()))

    # train the model
    model.fit(
        x=ds_train,
        steps_per_epoch=1281167 // BATCH_SIZE,
        validation_data=ds_validation,
        validation_steps=50000 // BATCH_SIZE,
        callbacks=[model_checkpoint, tensorboard],
        # The following doesn't seem to help.
        # use_multiprocessing=True, workers=4,
        epochs=EPOCHS)

    # save the trained model
    model.save('model-final.h5')


if __name__ == '__main__':
    main()
