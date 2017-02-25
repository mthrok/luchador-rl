"""Implement Layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging

from theano.tensor.signal.pool import pool_2d

from .. import wrapper

__all__ = ['Pool2D']
_LG = logging.getLogger(__name__)
# pylint: disable=no-member


def _get_strides(strides, kernel_size):
    if strides is None:
        return kernel_size
    return strides


def _get_pad(padding, kernel_shape):
    if padding in ['valid', 'VALID']:
        return (0, 0)
    if padding in ['same', 'SAME']:
        height, width = kernel_shape
        return (height//2, width//2)
    if isinstance(padding, int):
        return [padding, padding]
    return padding


def _get_mode(mode):
    if mode.lower() in ['average', 'mean']:
        return 'average_exc_pad'
    return mode


class Pool2D(object):
    """Implement Pool2D layer in Theano.

    See :any:`BasePool2D` for detail.
    """
    def _build(self, input_tensor):
        ws = (self.args['kernel_height'], self.args['kernel_width'])
        mode = _get_mode(self.args['mode'])
        stride = _get_strides(self.args['strides'], ws)
        pad = _get_pad(self.args['padding'], ws)

        _tensor = pool_2d(
            input_tensor.unwrap(), ws=ws, stride=stride, mode=mode,
            pad=pad, ignore_border=True)

        input_shape = input_tensor.shape
        new_height = input_tensor.shape[2] // self.args['kernel_height']
        new_width = input_tensor.shape[3] // self.args['kernel_width']

        output_shape = (input_shape[0], input_shape[1], new_height, new_width)

        return wrapper.Tensor(_tensor, shape=output_shape, name='output')
