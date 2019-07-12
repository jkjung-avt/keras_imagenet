keras_imagenet
==============

This repository contains code I use to train ImageNet (ILSVRC2012) image classification models from scratch.

**Highlight #1**: I use [TFRecords](https://www.tensorflow.org/tutorials/load_data/tf_records) and [tf.data.TFRecordDataset API](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) to speed up data ingestion of the training pipeline.  This way I could multi-process the data pre-processing (and online data augmentation) task, and keep the GPUs maximally utilized.

**Highlight #2**: In addition to heavy data augmentation, I also use various tricks in attempt to achieve best accuracy for the trained image classification models.  More specifically, I implement 'iter_size' and use 'AdamW' (Adam optimizer with decoupled weight decay) in the code.

I took most of the dataset preparation code from tensorflow [models/research/inception](https://github.com/tensorflow/models/tree/master/research/inception).  It was under Apache license as specified [here](https://github.com/tensorflow/models/blob/master/LICENSE).

Otherwise, please refer to the following blog post of mine for some more  implementation details about the code:

[To be updated...](https://jkjung-avt.github.io/)

# Prerequisite

The dataset and CNN models in this repository are built and trained using the keras API within tensorflow.  To use the code within, make sure you have a relatively new version of tensorflow (say, 1.9.0+) installed properly on the system before running the code.  I myself have tested it with tensorflow 1.11.0, 1.12.2 and 1.14.0.

In addition, the python code in this repository is for python3.  Make sure you have tensorflow and its dependencies working for python3.

# Step-by-step

1. Download the 'Training images (Task 1 & 2)' and 'Validation images (all tasks)' from the [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) download page](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads).

   ```shell
   $ ls -l ${HOME}/Downloads/
   -rwxr-xr-x 1 jkjung jkjung 147897477120 Nov  7  2018 ILSVRC2012_img_train.tar
   -rwxr-xr-x 1 jkjung jkjung   6744924160 Nov  7  2018 ILSVRC2012_img_val.tar
   ```

2. Untar the 'train' and 'val' files.  For example, I put the untarred files at ${HOME}/data/ILSVRC2012/.

   ```shell
   $ mkdir -p ${HOME}/data/ILSVRC2012
   $ cd ${HOME}/data/ILSVRC2012
   $ mkdir train
   $ cd train
   $ tar xvf ${HOME}/Downloads/ILSVRC2012_img_train.tar
   $ find . -name "*.tar" | while read NAME ; do \
         mkdir -p "${NAME%.tar}"; \
         tar -xvf "${NAME}" -C "${NAME%.tar}"; \
         rm -f "${NAME}"; \
     done
   $ cd ..
   $ mkdir validation
   $ cd validation
   $ tar xvf ${HOME}/Downloads/ILSVRC2012_img_val.tar
   ```

3. Clone this repository.

   ```shell
   $ cd ${HOME}/project
   $ git clone https://github.com/jkjung-avt/keras_imagenet.git
   $ cd keras_imagenet
   ```

4. Pre-process the validation image files (moving the JPEG files into corresponding subfolders).

   ```shell
   $ cd data
   $ python3 ./preprocess_imagenet_validation_data.py \
             ${HOME}/data/ILSVRC2012/validation \
             imagenet_2012_validation_synset_labels.txt
   ```

5. Build TFRecord files for 'train' and 'validation'.  (This step could takes a couple of hours, since there are 1,281,167 training images and 50,000 validation images in total.)

   ```shell
   $ mkdir ${HOME}/data/ILSVRC2012/tfrecords
   $ python3 build_imagenet_data.py \
             --output_directory ${HOME}/data/ILSVRC2012/tfrecords \
             --train_directory ${HOME}/data/ILSVRC2012/train \
             --validation_directory ${HOME}/data/ILSVRC2012/validation
   ```

6. As an example, train a MobileNetV2 model.  Take a peek at the `train_mobilenet_v2.sh` script before running it.  You could certainly modify the learning rate schedule, weight decay and epochs in the script to see if you could try a model with better accuracy.  (On my desktop PC with an NVIDIA GTX-1080 Ti, it takes roughly 2 weeks to train this model for 200 epochs.)

   ```shell
   $ ./train_mobilenet_v2.sh
   ```

   Here is a list of options for the `train.py` script:

   * `--batch_size`: batch size for both training and validation
   * `--iter_size`: aggregating gradients before doing 1 weight update
   * `--initial_lr`: initial learning rate
   * `--final_lr`: final learning rate (learning rate is decreased linearly for each epoch)
   * `--weight_decay`: weight decay value for 'AdamW'
   * `--epochs`: total number of epochs

   **Additional Notes on MobileNetV2**

   For some reason, Keras has trouble loading a trained/saved MobileNetV2 model.  The load_model() call would fail with error message:
   
      `TypeError: '<' not supported between instances of 'dict' and 'float'`
   
   To work around this problem, I followed [this post](https://github.com/tensorflow/tensorflow/issues/22697#issuecomment-436301471) and added the following at line 309 (after the `super()` call of `ReLU`) lines in `/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/layers/advanced_activations.py`.
   
      ```python
          if type(max_value) is dict:
              max_value = max_value['value']
          if type(negative_slope) is dict:
              negative_slope = negative_slope['value']
          if type(threshold) is dict:
              threshold = threshold['value']
      ```
   
7. Evaluate accuracy of the trained MobileNetv2 model.

   ```shell
   $ python3 evaluate.py --dataset_dir ${HOME}/data/ILSVRC2012/tfrecords \
                         saves/mobilenet_v2-model-final.h5
   ```

8. For training other CNN models, check out `models/models.py`.  In addition to `mobilenet_v2`, `nasnet_mobile` and `resnet50`, you could also implement your own Keras CNN models by extending the code.
