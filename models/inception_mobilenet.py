"""Inception MobileNet (224x224) model for Keras.

The InceptionMobileNet model is a combination of the designs of
"InceptionV2", "InceptionV3" and "MobileNetV1", as an attempt to
create a fast and good feature extractor for object detection
and image segmentation applications.

Reference:
1. inception_v2.py in this repository.
2. https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
3. https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils


def _conv2d_bn(x,
               filters,
               kernel_size=(3, 3),
               padding='same',
               strides=(1, 1),
               name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size of the convolution.
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


def _depthwise_conv2d_bn(x,
                         filters,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=(1, 1),
                         name=None):
    """Utility function to apply factorized (depthwise & pointwise) conv + BN.

    # Arguments
        x: input tensor.
        filters: number of (pointwise) output channels.
        kernel_size: kernel size of the (depthwise) convolution.
        padding: padding mode of the depthwise convolution.
        strides: strides of the (depthwise) convolution.
        name: name of the ops; will become
              `name + '_dw_conv'` for the depthwise convolution,
              `name + '_dw_bn'` for the depthwise batch norm layer,
              `name + '_dw_relu'` for the depthwise relu layer,
              `name + '_pw_conv'` for the pointwise convolution,
              `name + '_pw_bn'` for the pointwise batch norm layer,

    # Returns
        Output tensor after applying the factorized conv + BN.
    """
    if name is not None:
        dw_conv_name = name + '_dw_conv'
        dw_bn_name = name + '_dw_bn'
        dw_relu_name = name + '_dw_relu'
        pw_conv_name = name + '_pw_conv'
        pw_bn_name = name + '_pw_bn'
    else:
        dw_conv_name, dw_bn_name, dw_relu_name = None, None, None
        pw_conv_name, pw_bn_name = None, None
    bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=dw_conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=dw_bn_name)(x)
    x = layers.Activation('relu', name=dw_relu_name)(x)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        name=pw_conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=pw_bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def _mixed(x, filters, name=None):
    """Utility function to implement the mixed (inception mobilenet) block.

    # Arguments
        x: input tensor.
        filters: a list of filter sizes.
        name: name of the ops

    # Returns
        Output tensor after applying the mixed block.
    """
    if len(filters) != 4:
        raise ValueError('filters should have 4 components')

    name1 = name + '_1x1' if name else None
    branch1x1 = _conv2d_bn(x, filters[0],
                           kernel_size=(1, 1),
                           name=name1)

    name1 = name + '_3x3' if name else None
    branch3x3 = _depthwise_conv2d_bn(x, filters[1],
                                     kernel_size=(3, 3),
                                     name=name1)

    name1 = name + '_5x5' if name else None
    branch5x5 = _depthwise_conv2d_bn(x, filters[2],
                                     kernel_size=(5, 5),
                                     name=name1)

    name1 = name + '_7x7' if name else None
    branch7x7 = _depthwise_conv2d_bn(x, filters[3],
                                     kernel_size=(7, 7),
                                     name=name1)

    concat_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.concatenate(
        [branch1x1, branch3x3, branch5x5, branch7x7],
        axis=concat_axis,
        name=name)
    return x


def _mixed_s2(x, filters, name=None):
    """Utility function to implement the 'stride-2' mixed block.

    # Arguments
        x: input tensor.
        filters: a list of filter sizes.
        name: name of the ops

    # Returns
        Output tensor after applying the 'stride-2' mixed block.
    """
    if len(filters) != 2:
        raise ValueError('filters should have 2 components')

    name1 = name + '_3x3' if name else None
    branch3x3 = _depthwise_conv2d_bn(x, filters[0],
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     name=name1)

    name1 = name + '_5x5' if name else None
    branch5x5 = _depthwise_conv2d_bn(x, filters[1],
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     name=name1)

    name1 = name + '_pool' if name else None
    branchpool = layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2),
                                     name=name1)(x)

    concat_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.concatenate(
        [branch3x3, branch5x5, branchpool],
        axis=concat_axis,
        name=name)
    return x


def InceptionMobileNet(include_top=False,
                       weights=None,
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       classes=1000,
                       **kwargs):
    """Instantiates the InceptionMobileNet architecture.

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

    x = _conv2d_bn(img_input, 32, (3, 3), strides=(2, 2),
                   name='conv1a')                          # 1a: 112x112x32
    x = _conv2d_bn(x, 32, (3, 3), name='conv1b')           # 1b: 112x112x32
    x = _conv2d_bn(x, 64, (3, 3), name='conv1c')           # 1c: 112x112x64

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                            name='pool2a')(x)              # 2a: 56x56x64
    x = _conv2d_bn(x,  80, (1, 1), name='conv2b')          # 2b: 56x56x80
    x = _conv2d_bn(x, 128, (3, 3), name='conv2c')          # 2c: 56x56x128

    x = _mixed_s2(x, (96, 32), name='mixed3a_s2')          # 3a: 28x28x256
    x = _mixed(x,  ( 64,  64,  96,  32), name='mixed3b')   # 3b: 28x28x256
    x = _mixed(x,  ( 64,  64,  96,  64), name='mixed3c')   # 3c: 28x28x288
    x = _mixed(x,  ( 64,  96,  96,  64), name='mixed3d')   # 3d: 28x28x320
    x = _mixed(x,  ( 64,  96,  96,  64), name='mixed3e')   # 3e: 28x28x320

    x = _mixed_s2(x, (240, 80), name='mixed4a_s2')         # 4a: 14x14x640
    x = _mixed(x,  (128, 192, 192, 128), name='mixed4b')   # 4b: 14x14x640
    x = _mixed(x,  (128, 192, 192, 128), name='mixed4c')   # 4c: 14x14x640

    x = _mixed_s2(x, (480, 160), name='mixed5a_s2')        # 5a: 7x7x1280
    x = _mixed(x,  (256, 384, 384, 256), name='mixed5b')   # 5b: 7x7x1280
    x = _mixed(x,  (256, 384, 384, 256), name='mixed5c')   # 5c: 7x7x1280

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
    model = models.Model(inputs, x, name='inception_mobilenet')

    return model
