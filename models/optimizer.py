"""optimizer.py

Code modified from:
https://github.com/keras-team/keras/issues/3556#issuecomment-490375678

NOTE: This code does not handle difference in lr 'decay' when iter_size
is set to some value greater than 1.  As a workaround, try to adjust
'decay' lower in the original optimizer...
"""


import tensorflow as tf
from tensorflow.keras import backend as K


def convert_to_accum_optimizer(orig_optimizer, iter_size, do_mean=True):
    """Convert a Keras optimizer to make it support 'iter_size'

    # Args
        orig_optimizer: the original optimizer
        iter_size: accumulate gradients for this meany iterations
        do_mean: use mean (average) grandient instead of sum
    """
    if K.backend() != 'tensorflow':
        raise RuntimeError('convert_to_accum_optimizer() only supports '
                           'tensorflow backend!')
    if iter_size < 1:
        raise ValueError('iter_size must be >= 1')
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    orig_optimizer.iter_size = K.variable(iter_size, dtype='int64')
    orig_optimizer.do_mean = do_mean

    def new_get_gradients(self, loss, params):
        return self.accumulated_grads

    def new_get_updates(self, loss, params):
        if not hasattr(self, 'accumulated_grads'):
            self.accumulated_grads = [
                K.zeros(K.int_shape(p), dtype=K.dtype(p))
                for p in params]
        new_grads = orig_get_gradients(loss, params)
        if self.do_mean:
            new_grads = [
                g / K.cast(self.iter_size, K.dtype(g)) for g in new_grads]
        update_grads = [
            K.update_add(p, g)
            for p, g in zip(self.accumulated_grads, new_grads)]

        def update_func():
            with tf.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [
                    K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p)))
                    for p in self.accumulated_grads]
            return tf.group(*reset_grads)

        def empty_func():
            return tf.group([])

        # call the original optimizer's get_updates() function only
        # once every 'iter_size' iterations
        update_switch = K.equal(self.iterations % self.iter_size,
                                self.iter_size - 1)
        with tf.control_dependencies(update_grads):
            self.updates = [
                K.switch(update_switch, update_func, empty_func)]
            return self.updates

    orig_optimizer.get_gradients = new_get_gradients.__get__(
        orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = new_get_updates.__get__(
        orig_optimizer, type(orig_optimizer))
    return orig_optimizer
