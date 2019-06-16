"""optimizer.py

Code modified from:
https://github.com/keras-team/keras/issues/3556#issuecomment-490375678
"""


import tensorflow as tf
from tensorflow.keras import backend as K


def convert_to_accum_optimizer(orig_optimizer, iter_size, do_mean=False):
    """Convert a Keras optimizer to make it support 'iter_size'

    # Args
        orig_optimizer: the original optimizer
        iter_size: accumulate gradients for this meany iterations
        do_mean: use mean (average) grandient instead of sum
    """
    if K.backend != 'tensorflow':
        raise TypeError('convert_to_accum_optimizer() only supports'
                        'tensorflow backend!')
    if iter_size < 1:
        raise ValueError('iter_size must be >= 1')
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iters = K.variable(0, dtype='int64', name='accumulated_iters')
    orig_optimizer.accumulated_iters = accumulated_iters

    def new_get_gradients(self, loss, params):
        return self.accumulated_grads

    def new_get_updates(self, loss, params):
        self.accumulated_grads = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p))
            for p in params]
        iters = K.update_add(self.accumulated_iters, 1)
        new_grads = orig_get_gradients(loss, params)
        if do_mean:
            new_grads = [
                g / K.cast(iter_size, K.dtype(g)) for g in new_grads]
        self.updated_grads = [
            K.update_add(p, g)
            for p, g in zip(self.accumulated_grads, new_grads)]

        def update_function():
            with tf.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [
                    K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p)))
                    for p in self.accumulated_grads]
            return tf.group(*(reset_grads + [iters]))

        def just_store_function():
            return tf.group(*[iters])

        update_switch = K.equal(iters % iter_size, 0)
        with tf.control_dependencies(self.updated_grads):
            self.updates = [
                K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = new_get_gradients.__get__(
        orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = new_get_updates.__get__(
        orig_optimizer, type(orig_optimizer))
    return orig_optimizer
