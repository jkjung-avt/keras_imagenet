Optimize Trained Models With TensorRT
=====================================

This article documents how I optimized my trained tf.keras models with TensorRT and verify inference performance of the resulting TensorRT engine.  I've verified the following precedure works on both a x86_64 PC and the Jetson platforms.

Refer to my blog post for some additional details: [Applying TensorRT on My tf.keras ImageNet Models](https://jkjung-avt.github.io/trt-keras-imagenet/).

# Reference:

* [Speeding up Deep Learning Inference Using TensorFlow, ONNX, and TensorRT](https://devblogs.nvidia.com/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)

# Prerequisite

* Install TensorRT.

    - For x86_64 PC's, refer to NVIDIA's official documentation: [Installation Guide :: NVIDIA Deep Learning TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).  Note that you should also install the **python3 binding** of TensorRT.
    - For the Jetson platforms, make sure to install one of the newer versions of [JetPack SDK's](https://developer.nvidia.com/embedded/jetpack) on the system.  Or you could refer to my blog posts for more details:
        - [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/)
        - [JetPack-4.3 for Jetson Nano](https://jkjung-avt.github.io/jetpack-4.3/)
        - [JetPack-4.4 for Jetson Nano](https://jkjung-avt.github.io/jetpack-4.4/)

* Install "tf2onnx" for python3.

    ```shell
    $ pip3 install --user onnxruntime
    $ pip3 install --user tf2onnx
    ```

* Install "PyCUDA" for python3.

    - You could use my [install_pycuda.sh](https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/install_pycuda.sh) script.
    - Or refer to the [official documentation](https://wiki.tiker.net/PyCuda/Installation/).

# Step-by-step:

The following uses a trained tf.keras "googlenet_bn" model as example.  (As stated in my [keras_imagenet/README.md](https://github.com/jkjung-avt/keras_imagenet/blob/master/README.md), you could download my pre-trained .h5 files from my GoogleDrive links.)

1. Convert the tf.keras model to a frozen inference graph (.pb).

    ```shell
    $ cd ${HOME}/project/keras_imagenet/tensorrt
    $ PYTHONPATH=.. \
      python3 tf_h5_to_pb.py ../saves/googlenet_bn-acc_0.7091.h5 \
                             googlenet_bn.pb
    ......
    input tensor:  input_1
    output tensors:  ['Logits/Softmax']
    ```

2. Convert the .pb file to ONNX.

    ```shell
    $ python3 -m tf2onnx.convert --input googlenet_bn.pb \
                                 --inputs input_1:0 \
                                 --outputs Logits/Softmax:0 \
                                 --output googlenet_bn.onnx
    ......
    2020-06-26 09:52:41,637 - INFO - Using tensorflow=1.14.0, onnx=1.4.1, tf2onnx=1.5.6/80edd7
    2020-06-26 09:52:41,637 - INFO - Using opset <onnx, 8>
    2020-06-26 09:52:42,820 - INFO - Optimizing ONNX model
    2020-06-26 09:52:45,043 - INFO - After optimization: Const -10 (297->287), Identity -2 (2->0), Transpose -252 (254->2)
    2020-06-26 09:52:45,085 - INFO -
    2020-06-26 09:52:45,085 - INFO - Successfully converted TensorFlow model googlenet_bn.pb to ONNX
    2020-06-26 09:52:45,318 - INFO - ONNX model is saved at googlenet_bn.onnx
    ```

3. (Optional) Use "trtexec" to convert ONNX to TensorRT and measure inference time.

    For TensorRT 6: (no "--explicitBatch")

    ```shell
    $ ${TRT_BIN}/trtexec --onnx=googlenet_bn.onnx \
                         --fp16 \
                         --workspace=1024 \
                         --warmUp=2 \
                         --dumpProfile \
                         --verbose \
                         --saveEngine=googlenet_bn.engine
    ```

    Or for TensorRT 7+: (The following was measured on my x86_64 PC with GeForce RTX 2080 Ti and TensorRT 7.0.0.11.)

    ```shell
    $ ${TRT_BIN}/trtexec --onnx=googlenet_bn.onnx \
                         --explicitBatch \
                         --shapes=input_1:0:1x224x224x3 \
                         --fp16 \
                         --workspace=1024 \
                         --warmUp=2 \
                         --dumpProfile \
                         --verbose \
                         --saveEngine=googlenet_bn.engine
    ......
    ----------------------------------------------------------------
    Input filename:   googlenet_bn.onnx
    ONNX IR version:  0.0.4
    Opset version:    8
    Producer name:    tf2onnx
    Producer version: 1.5.6
    Domain:
    Model version:    0
    Doc string:
    ----------------------------------------------------------------
    ......
    [06/26/2020-09:53:46] [W] Dynamic dimensions required for input: input_1:0, but
    no shapes were provided. Automatically overriding shape to: 1x224x224x3
    ......
    [06/26/2020-09:56:53] [I] GPU Compute
    [06/26/2020-09:56:53] [I] min: 0.656097 ms
    [06/26/2020-09:56:53] [I] max: 27.9177 ms
    [06/26/2020-09:56:53] [I] mean: 0.742242 ms
    [06/26/2020-09:56:53] [I] median: 0.663818 ms
    [06/26/2020-09:56:53] [I] percentile: 1.65561 ms at 99%
    ......
    (Per layer statistics omitted)
    ```

4. This step could be skipped if the "googlenet_bn.engine" file has been generated by the previous step.  Otherwise, use "build_engine.py" to convert the ONNX to a TensorRT engine.  For debugging, you could use the "-v" or "--verbose" command-line option to enable verbose logs.

    ```shell
    $ python3 build_engine.py googlenet_bn.onnx googlenet_bn.engine
    [TensorRT] WARNING: onnx2trt_utils.cpp:198: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    [TensorRT] WARNING: onnx2trt_utils.cpp:198: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    ```

5. Compare inference results between the original tf.keras model and the optimized TensorRT engine.  Note that you'll need to have "PyCUDA" installed for this to work.

    I used this [huskies.jpg](https://raw.githubusercontent.com/jkjung-avt/tf_trt_models/master/examples/detection/data/huskies.jpg) picture for testing.

    ```
    $ cd ${HOME}/project/keras_imagenet
    $ wget https://raw.githubusercontent.com/jkjung-avt/tf_trt_models/master/examples/detection/data/huskies.jpg
    ### First test with the tf.keras model
    $ python3 predict_image.py saves/googlenet_bn-acc_0.7091.h5 \
                               huskies.jpg
    0.96   n03218198 dogsled, dog sled, dog sleigh
    0.03   n02109961 Eskimo dog, husky
    0.00   n02110185 Siberian husky
    0.00   n02110063 malamute, malemute, Alaskan malamute
    0.00   n02114367 timber wolf, grey wolf, gray wolf, Canis lupus
    ### Then run the TensorRT engine and compare
    $ python3 predict_image.py tensorrt/googlenet_bn.engine \
                               huskies.jpg
    0.96   n03218198 dogsled, dog sled, dog sleigh
    0.03   n02109961 Eskimo dog, husky
    0.00   n02110185 Siberian husky
    0.00   n02110063 malamute, malemute, Alaskan malamute
    0.00   n02114367 timber wolf, grey wolf, gray wolf, Canis lupus
    ```
