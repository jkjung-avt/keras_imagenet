"""tf_h5_to_pb.py

For converting a tf.keras (.h5) model to a frozen inference (.pb) file.  Refernce: https://devblogs.nvidia.com/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/

Example usage:
$ PYTHONPATH=..:$PYTHONPATH \
  python3 tf_h5_to_pb.py ../saves/googlenet_bn-acc_0.7091.h5 googlenet_bn.pb
"""


import argparse

import tensorflow as tf

from utils.utils import config_keras_backend, clear_keras_session
from models.adamw import AdamW


def keras_to_pb(model, output_filename, output_node_names):
   """
   This is the function to convert the Keras model to pb.

   Args:
      model: The Keras model.
      output_filename: The output .pb file name.
      output_node_names: The output nodes of the network. If None, then
          the function gets the last layer name as the output node.
   """

   # Get the names of the input and output nodes.
   in_name = model.layers[0].get_output_at(0).name.split(':')[0]

   if output_node_names is None:
       output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

   sess = tf.keras.backend.get_session()

   # The TensorFlow freeze_graph expects a comma-separated string of
   # output node names.
   output_node_names_tf = ','.join(output_node_names)

   frozen_graph_def = tf.graph_util.convert_variables_to_constants(
       sess,
       sess.graph_def,
       output_node_names)

   wkdir = ''
   tf.train.write_graph(
        frozen_graph_def, wkdir, output_filename, as_text=False)

   return in_name, output_node_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5', type=str)
    parser.add_argument('pb', type=str)
    args = parser.parse_args()

    if not args.h5.endswith('.h5'):
        raise SystemExit('bad keras model file name (not .h5)')

    config_keras_backend()
    tf.keras.backend.set_learning_phase(0)

    model = tf.keras.models.load_model(
        args.h5, compile=False, custom_objects={'AdamW': AdamW})

    in_tensor_name, out_tensor_names = keras_to_pb(
        model, args.pb, None)
    print('input tensor: ', in_tensor_name)
    print('output tensors: ', out_tensor_names)

    clear_keras_session()


if __name__ == '__main__':
    main()
