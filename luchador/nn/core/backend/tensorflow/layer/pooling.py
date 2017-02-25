"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

import luchador
from .. import wrapper

__all__ = ['Pool2D']
_LG = logging.getLogger(__name__)
# pylint: disable=no-member


def _get_format(data_format):
    return data_format or luchador.get_nn_conv_format()


def _get_kernel_size(kernel_height, kernel_width, data_format):
    if data_format == 'NHWC':
        return (1, kernel_height, kernel_width, 1)
    return (1, 1, kernel_height, kernel_width)


def _get_strides(strides, kernel_size, data_format):
    if isinstance(strides, int):
        strides = [strides, strides]

    if strides is None:
        strides = kernel_size

    if len(strides) == 4:
        return strides

    if data_format == 'NHWC':
        return [1, strides[0], strides[1], 1]

    return [1, 1, strides[0], strides[1]]


class Pool2D(object):
    """Implement Pool2D layer in Theano.

    See :any:`Pool2D` for detail.
    """
    def _build(self, input_tensor):
        data_format = _get_format(self.args.get('data_format'))
        ksize = _get_kernel_size(
            self.args['kernel_height'], self.args['kernel_width'], data_format)
        strides = _get_strides(self.args['strides'], ksize, data_format)

        padding = self.args['padding'].upper()

        if self.args['mode'] == 'max':
            func = tf.nn.max_pool
        elif self.args['mode'] == 'average':
            func = tf.nn.avg_pool

        _tensor = func(
            input_tensor.unwrap(), ksize=ksize, strides=strides,
            padding=padding, data_format=data_format)

        return wrapper.Tensor(_tensor, name='output')
