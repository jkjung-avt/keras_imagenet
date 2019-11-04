"""optimizer.py

Code modified from:
https://github.com/keras-team/keras/issues/3556#issuecomment-490375678

NOTE: This code most likely does not work for keras OptimizerV2 (in
tensorflow 1.13.0+).
"""


import tensorflow as tf
from tensorflow.keras import backend as K


def convert_to_accum_optimizer(optimizer, iter_size, do_mean=True):
    """Convert a Keras optimizer to make it support 'iter_size'

    # Args
        optimizer: the original optimizer
        iter_size: accumulate gradients for this meany iterations
        do_mean: use mean (average) grandient instead of sum
    """
    ver = tf.__version__.split('.')
    if int(ver[0]) >= 2 or int(ver[1]) >= 13:
        raise RuntimeError('convert_to_accum_optimizer() only supports '
                           'tensorflow version <= 1.12.x!')
    if iter_size < 1:
        raise ValueError('iter_size must be >= 1')
    if hasattr(optimizer, 'iter_size'):
        raise RuntimeError('optimizer already has iter_size!')
    optimizer.orig_get_gradients = optimizer.get_gradients
    optimizer.orig_get_updates = optimizer.get_updates
    optimizer.iter_size = K.variable(
        iter_size, dtype='int64', name='iter_size')
    optimizer.do_mean = do_mean
    optimizer.accumulated_iterations = K.variable(
        0, dtype='int64', name='accumulated_iterations')

    def new_get_gradients(self, loss, params):
        return self.accumulated_grads

    def new_get_updates(self, loss, params):
        self.accumulated_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p))
                                  for p in params]
        update_iter = [K.update_add(self.accumulated_iterations, 1)]
        new_grads = self.orig_get_gradients(loss, params)
        if self.do_mean:
            new_grads = [g / K.cast(self.iter_size, K.dtype(g))
                         for g in new_grads]
        update_grads = [K.update_add(p, g)
                        for p, g in zip(self.accumulated_grads, new_grads)]

        def update_func():
            with tf.control_dependencies(self.orig_get_updates(loss, params)):
                reset_grads = [
                    K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p)))
                    for p in self.accumulated_grads]
            return tf.group(*(reset_grads + update_iter))

        def just_iter_func():
            return tf.group(*update_iter)

        # do the original get_updates() computations only once every
        # 'iter_size' iterations
        update_switch = K.equal(
            self.accumulated_iterations % self.iter_size, 0)
        with tf.control_dependencies(update_grads):
            self.updates = [
                K.switch(update_switch, update_func, just_iter_func)]
            return self.updates

    # convert new_get_gradients() and new_get_updates() to class methods
    optimizer.get_gradients = new_get_gradients.__get__(
        optimizer, type(optimizer))
    optimizer.get_updates = new_get_updates.__get__(
        optimizer, type(optimizer))
    return optimizer
