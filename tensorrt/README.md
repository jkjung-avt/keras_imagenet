Optimize Trained Models With TensorRT
=====================================

# Reference:

* [Speeding up Deep Learning Inference Using TensorFlow, ONNX, and TensorRT](https://devblogs.nvidia.com/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)

# Prerequisite

* Install TensorRT.

* Install "tf2onnx" for python3.

    ```shell
    $ pip3 install --user onnxruntime
    $ pip3 install --user tf2onnx
    ```

# Step-by-step:

The following uses a trained tf.keras "googlenet_bn" model as example.

1. Convert the tf.keras model to a frozen inference graph (.pb).

    ```shell
    $ cd ${HOME}/project/keras_imagenet/tensorrt
    $ PYTHONPATH=.. \
      python3 tf_h5_to_pb.py ../saves/googlenet_bn-acc_0.7091.h5 \
                             googlenet_bn.pb
    ```

2. Convert the .pb file to ONNX.

    ```shell
    $ python3 -m tf2onnx.convert --input googlenet_bn.pb \
                                 --inputs input_1:0 \
                                 --outputs Logits/Softmax:0 \
                                 --output googlenet_bn.onnx
    ```

3. Use "trtexec" to convert ONNX to TensorRT and measure inference time.  NOTE: The "--explicitBatch" is specific to TensorRT 7.0+.

    ```shell
    $ ${TRT_BIN}/trtexec --onnx=googlenet_bn.onnx \
                         --explicitBatch \
                         --fp16 \
                         --workspace=1024 \
                         --warmUp=2 \
                         --iterations=1000 \
                         --dumpProfile \
                         --verbose \
                         --saveEngine=googlenet_bn.engine
    ```
