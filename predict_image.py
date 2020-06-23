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
                        help='e.g. saves/googlenet_bn-model-final.h5')
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


def infer_with_trt(img, model):
    """Inference the image with TensorRT engine."""
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings = [], [], [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    stream = cuda.Stream()
    context = engine.create_execution_context()
    np.copyto(host_inputs[0], img.ravel())
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    return host_outputs[0]


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
