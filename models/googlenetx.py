"""GoogLeNetX model for Keras.

The code is adapted from keras_applications' InceptionV3 code.  This
'GoogLeNetX' implementation differs from the original GoogLeNet in
several ways:

1. Number of filters are different in later layers.
2. Ratio between 1x1, 3x3 and 5x5 filters is not the same.
3. LRN layers are replaced by BatchNormalization layers.

NOTE: This model is still experimental.  I expect to keep modifying
it for a while.
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
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def inception(x,
              nb_filters):
    """Utility function to implement the inception module.

    # Arguments
        x: input tensor.
        nb_filters: number of output filters.

    # Returns
        Output tensor after applying the inception.
    """
    if nb_filters % 16 != 0:
        raise ValueError('nb_filters must be a multiple of 16')
    branch1x1 = conv2d_bn(x, nb_filters // 4, (1, 1))

    branch3x3 = conv2d_bn(x, nb_filters // 4, (1, 1))
    branch3x3 = conv2d_bn(branch3x3, nb_filters // 2, (3, 3))

    branch5x5 = conv2d_bn(x, nb_filters // 16, (1, 1))
    branch5x5 = conv2d_bn(branch5x5, nb_filters // 8, (5, 5))

    branchpool = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = conv2d_bn(branchpool, nb_filters // 8, (1, 1))

    if backend.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = 3
    x = layers.concatenate(
        [branch1x1, branch3x3, branch5x5, branchpool], axis=concat_axis)
    return x


def GoogLeNetX(include_top=False,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               **kwargs):
    """Instantiates the GoogLeNetX architecture.

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

    x = conv2d_bn(img_input, 64, (7, 7), strides=(2, 2))
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = conv2d_bn(x, 192, (3, 3))
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, 256)   # inception3a: 28x28x256
    x = inception(x, 480)   # inception3b: 28x28x480
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, 512)   # inception4a: 14x14x512
    x = inception(x, 512)   # inception4b: 14x14x512
    x = inception(x, 512)   # inception4c: 14x14x512
    x = inception(x, 528)   # inception4d: 14x14x528
    x = inception(x, 832)   # inception4e: 14x14x832
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, 832)   # inception5a: 7x7x832
    x = inception(x, 1024)  # inception5b: 7x7x102

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='googlenetx')

    return model
