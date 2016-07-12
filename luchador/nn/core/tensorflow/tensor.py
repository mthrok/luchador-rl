from __future__ import absolute_import

import tensorflow as tf

from ..base.tensor import Tensor as BaseTensor


class Tensor(BaseTensor):
    def __init__(self, tensor, shape=None, name=None):
        if tensor is not None:
            name = name or tensor.name
            shape = shape or tensor.get_shape().as_list()
        super(Tensor, self).__init__(tensor=tensor, shape=shape, name=name)


class Input(Tensor):
    def __init__(self, dtype=tf.float32, shape=None, name=None):
        super(Input, self).__init__(tensor=None, shape=shape, name=name)

        self.dtype = dtype

    def __call__(self):
        return self.build()

    def build(self):
        if self.tensor is None:
            self.tensor = tf.placeholder(
                dtype=self.dtype, shape=self.shape, name=self.name)
        return self