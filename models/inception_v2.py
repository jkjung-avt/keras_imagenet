"""Inception V2 (224x224) model for Keras.

Reference: (from tensorflow 'slim' library)
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py

This implementation differs from the original one:
    * Not using SeparableConv2D in '1a'
    * AveragePooling2D in inception module '5c'
    * BatchNorm placement?
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils


def conv2d_bn(x,
              filters,
              kernel_size=(3, 3),
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size of the convolution
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def inception(x, filters):
    """Utility function to implement the inception module.

    # Arguments
        x: input tensor.
        filters: a list of filter sizes.

    # Returns
        Output tensor after applying the inception.
    """
    if len(filters) != 4:
        raise ValueError('filters should have 4 components')
    if len(filters[1]) != 2 or len(filters[2]) != 2:
        raise ValueError('incorrect spec of filters')

    branch1x1 = conv2d_bn(x, filters[0], (1, 1))

    branch3x3 = conv2d_bn(x, filters[1][0], (1, 1))
    branch3x3 = conv2d_bn(branch3x3, filters[1][1], (3, 3))

    # branch5x5 is implemented with two 3x3 conv2d's
    branch5x5 = conv2d_bn(x, filters[2][0], (1, 1))
    branch5x5 = conv2d_bn(branch5x5, filters[2][1], (3, 3))
    branch5x5 = conv2d_bn(branch5x5, filters[2][1], (3, 3))

    # use AveragePooling2D here
    branchpool = layers.AveragePooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = conv2d_bn(branchpool, filters[3], (1, 1))

    concat_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.concatenate(
        [branch1x1, branch3x3, branch5x5, branchpool], axis=concat_axis)
    return x


def inception_s2(x, filters):
    """Utility function to implement the 'stride-2' inception module.

    # Arguments
        x: input tensor.
        filters: a list of filter sizes.

    # Returns
        Output tensor after applying the 'stride-2' inception.
    """
    if len(filters) != 2:
        raise ValueError('filters should have 2 components')
    if len(filters[0]) != 2 or len(filters[1]) != 2:
        raise ValueError('incorrect spec of filters')

    branch3x3 = conv2d_bn(x, filters[0][0], (1, 1))
    branch3x3 = conv2d_bn(branch3x3, filters[0][1], (3, 3), strides=(2, 2))

    branch5x5 = conv2d_bn(x, filters[1][0], (1, 1))
    branch5x5 = conv2d_bn(branch5x5, filters[1][1], (3, 3))
    branch5x5 = conv2d_bn(branch5x5, filters[1][1], (3, 3), strides=(2, 2))

    # use MaxPooling2D here
    branchpool = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    concat_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.concatenate(
        [branch3x3, branch5x5, branchpool], axis=concat_axis)
    return x


def InceptionV2(include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """Instantiates the InceptionV2 architecture.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: must be None.
        input_tensor: Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: input tensor shape, which is used to create an
            input tensor if `input_tensor` is not specified.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `avg` means that global average pooling will be applied
                to the output of the last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights is not None:
        raise ValueError('weights is not currently supported')
    if input_tensor is None:
        if input_shape is None:
            raise ValueError('neither input_tensor nor input_shape is given')
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv2d_bn(img_input, 64, (7, 7), strides=(2, 2))  # 1a: 112x112x64

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='same')(x)            # 2a: 56x56x64
    x = conv2d_bn(x,  64, (1, 1))                         # 2b: 56x56x64
    x = conv2d_bn(x, 192, (3, 3))                         # 2c: 56x56x192

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='same')(x)            # 3a: 28x28x192
    x = inception(x, ( 64, ( 64,  64), ( 64,  96),  32))  # 3b: 28x28x256
    x = inception(x, ( 64, ( 64,  96), ( 64,  96),  64))  # 3c: 28x28x320

    x = inception_s2(x, ((128, 160), (64,  96)))          # 4a: 14x14x576
    x = inception(x, (224, ( 64,  96), ( 96, 128), 128))  # 4b: 14x14x576
    x = inception(x, (192, ( 96, 128), ( 96, 128), 128))  # 4c: 14x14x576
    x = inception(x, (160, (128, 160), (128, 160),  96))  # 4d: 14x14x576
    x = inception(x, ( 96, (128, 192), (160, 192),  96))  # 4e: 14x14x576

    x = inception_s2(x, ((128, 192), (192, 256)))         # 5a: 7x7x1024
    x = inception(x, (352, (192, 320), (160, 224), 128))  # 5b: 7x7x1024
    x = inception(x, (352, (192, 320), (192, 224), 128))  # 5c: 7x7x1024

    # NOTE: 'AveragePooling2D' in '5c' (was 'MaxPooling2D' in original slim)

    if include_top:
        # Classification block
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='global_pool')(x)
        else:
            raise ValueError('bad spec of global pooling')
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_v2')

    return model


def InceptionV2X(include_top=False,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the InceptionV2X architecture.

    This model differs from InceptionV2 by moving 2 inception modules
    from 4x to 3x.

    # Returns
        A Keras model instance.
    """
    if weights is not None:
        raise ValueError('weights is not currently supported')
    if input_tensor is None:
        if input_shape is None:
            raise ValueError('neither input_tensor nor input_shape is given')
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv2d_bn(img_input, 64, (7, 7), strides=(2, 2))  # 1a: 112x112x64

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='same')(x)            # 2a: 56x56x64
    x = conv2d_bn(x,  64, (1, 1))                         # 2b: 56x56x64
    x = conv2d_bn(x, 128, (3, 3))                         # 2c: 56x56x128

    x = conv2d_bn(x, 192, (3, 3), strides=(2, 2))         # 3a: 28x28x256
    x = inception(x, ( 64, ( 64,  64), ( 64,  96),  32))  # 3b: 28x28x256
    x = inception(x, ( 64, ( 64,  80), ( 64,  96),  48))  # 3c: 28x28x288
    x = inception(x, ( 64, ( 64,  96), ( 64,  96),  64))  # 3d: 28x28x320
    x = inception(x, ( 64, ( 64, 128), ( 64, 128),  64))  # 3e: 28x28x384

    x = inception_s2(x, ((128, 160), (64,  96)))          # 4a: 14x14x576
    x = inception(x, (192, ( 96, 128), ( 96, 128), 128))  # 4b: 14x14x576
    x = inception(x, ( 96, (128, 192), (160, 192),  96))  # 4c: 14x14x576

    x = inception_s2(x, ((128, 192), (192, 256)))         # 5a: 7x7x1024
    x = inception(x, (352, (192, 320), (160, 224), 128))  # 5b: 7x7x1024
    x = inception(x, (352, (192, 320), (192, 224), 128))  # 5c: 7x7x1024

    if include_top:
        # Classification block
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='global_pool')(x)
        else:
            raise ValueError('bad spec of global pooling')
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_v2x')

    return model
