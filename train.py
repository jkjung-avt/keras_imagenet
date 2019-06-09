"""train.sh

This script is used to train the ImageNet models.

Example usage:
  $ python3 train.sh mobilenet_v2
"""


import time

import tensorflow as tf
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard


def main():
    # config tensorflow backend to use less GPU memory
    # Note: use CUDA_VISIBLE_DEVICES to control which GPU is used by Keras
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)

    model = MobileNetV2(include_top=True, weights=None, classes=1000)
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model_checkpoint = ModelCheckpoint(
        'chkpt-{epoch:03d}.h5', monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()))
    # train the model
    model.fit_generator(
        train_crops,
        steps_per_epoch = train_batches.samples // BATCH_SIZE,
        validation_data=valid_crops,
        validation_steps=valid_batches.samples // BATCH_SIZE,
        callbacks=[model_checkpoint, tensorboard],
        epochs = NUM_EPOCHS)
    
    # save trained weights
    model.save('model-final.h5')


if __name__ == '__main__':
    main()
