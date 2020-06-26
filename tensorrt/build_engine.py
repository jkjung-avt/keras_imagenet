"""build_engine.py

This script converts a ONNX model to an optimized TensorRT engine.

Example usage:
$ python3 build_engine.py googlenet_bn.onnx googlenet_bn.engine
"""


import os
import argparse

import tensorrt as trt


EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

MAX_BATCH = 1
FP16_MODE = True        # use FP32 if set to False
OUTPUT_TENSOR = 'prob'  # not used


def parse_args():
    """Parse command-line argements."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('onnx', type=str)
    parser.add_argument('engine', type=str)
    args = parser.parse_args()
    return args


def build_engine(onnx, verbose=False):
    """Build TensorRT engine from the ONNX model."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = MAX_BATCH
        builder.fp16_mode = FP16_MODE
        with open(onnx, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        if trt.__version__[0] >= '7':
            # set input to batch size 1
            shape = list(network.get_input(0).shape)
            shape[0] = 1
            network.get_input(0).shape = shape
        return builder.build_cuda_engine(network)


def main():
    args = parse_args()
    if not os.path.exists(args.onnx):
        raise SystemExit('ERROR: ONNX file (%s) not found!' % args.onnx)

    engine = build_engine(args.onnx, verbose=args.verbose)
    if engine:
        with open(args.engine, 'wb') as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    main()
