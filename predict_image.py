"""predict_image.py

This script is for testing a trained Keras ImageNet model.  The model
could be one of the following 2 formats:

    1. tf.keras model (.h5)
    2. optimized TensorRT engine (.engine)

Example usage #1:
$ python3 predict_image.py saves/googlenet_bn-model-final.h5 \
                           sample.jpg

Example usage #2:
$ python3 predict_image.py tensorrt/googlenet_bn.engine \
                           sample.jpg
"""


import argparse

import numpy as np
import cv2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help='a tf.keras model or a TensorRT engine, e.g. saves/googlenet_bn-model-final.h5 or tensorrt/googlenet_bn.engine')
    parser.add_argument('jpg',
                        help='an image file to be predicted')
    args = parser.parse_args()
    return args


def preprocess(img):
    """Preprocess an image for Keras ImageNet model inferencing."""
    if img.ndim != 3:
        raise TypeError('bad ndim of img')
    if img.dtype != np.uint8:
        raise TypeError('bad dtype of img')
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img *= (2.0/255)  # normalize to: 0.0~2.0
    img -= 1.0        # subtract mean to make it: -1.0~1.0
    img = np.expand_dims(img, axis=0)
    return img


def infer_with_tf(img, model):
    """Inference the image with TensorFlow model."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from utils.utils import config_keras_backend, clear_keras_session
    from models.adamw import AdamW

    config_keras_backend()

    # load the trained model
    net = tf.keras.models.load_model(model, compile=False,
                                     custom_objects={'AdamW': AdamW})
    predictions = net.predict(img)[0]

    clear_keras_session()

    return predictions


def init_trt_buffers(cuda, trt, engine):
    """Initialize host buffers and cuda buffers for the engine."""
    assert engine[0] == 'input_1:0'
    assert engine.get_binding_shape(0)[1:] == (224, 224, 3)
    size = trt.volume((1, 224, 224, 3)) * engine.max_batch_size
    host_input = cuda.pagelocked_empty(size, np.float32)
    cuda_input = cuda.mem_alloc(host_input.nbytes)
    assert engine[1] == 'Logits/Softmax:0'
    assert engine.get_binding_shape(1)[1:] == (1000,)
    size = trt.volume((1, 1000)) * engine.max_batch_size
    host_output = cuda.pagelocked_empty(size, np.float32)
    cuda_output = cuda.mem_alloc(host_output.nbytes)
    return host_input, cuda_input, host_output, cuda_output


def infer_with_trt(img, model):
    """Inference the image with TensorRT engine."""
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert len(engine) == 2, 'ERROR: bad number of bindings'
    host_input, cuda_input, host_output, cuda_output = init_trt_buffers(
        cuda, trt, engine)
    stream = cuda.Stream()
    context = engine.create_execution_context()
    context.set_binding_shape(0, (1, 224, 224, 3))
    np.copyto(host_input, img.ravel())
    cuda.memcpy_htod_async(cuda_input, host_input, stream)
    context.execute_async(
        batch_size=1,
        bindings=[int(cuda_input), int(cuda_output)],
        stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, cuda_output, stream)
    stream.synchronize()
    return host_output


def main():
    args = parse_args()

    # load the cls_list (index to class name)
    with open('data/synset_words.txt') as f:
        cls_list = sorted(f.read().splitlines())

    # load and preprocess the test image
    img = cv2.imread(args.jpg)
    if img is None:
        raise SystemExit('cannot load the test image: %s' % args.jpg)
    img = preprocess(img)

    # predict the image
    if args.model.endswith('.h5'):
        predictions = infer_with_tf(img, args.model)
    elif args.model.endswith('.engine'):
        predictions = infer_with_trt(img, args.model)
    else:
        raise SystemExit('ERROR: bad model')

    # postprocess
    top5_idx = predictions.argsort()[::-1][:5]  # take the top 5 predictions
    for i in top5_idx:
        print('%5.2f   %s' % (predictions[i], cls_list[i]))


if __name__ == '__main__':
    main()
