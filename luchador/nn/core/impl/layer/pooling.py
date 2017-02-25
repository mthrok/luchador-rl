"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer

__all__ = ['Pool2D']


class Pool2D(layer.Pool2D, BaseLayer):
    """Apply 2D Pooling

    Input Tensor : 4D tensor
        NCHW Format
            (batch size, **#input channels**, input height, input width)

        NHWC format : (Tensorflow backend only)
            (batch size, input height, input width, **#input channels**)

    Output Shape
        NCHW Format
            (batch size, **#output channels**, output height, output width)

        NHWC format : (Tensorflow backend only)
            (batch size, output height, output width, **#output channels**)

    Parameters
    ----------
    kernel_height, kernel : int
        Kernel shape (shape of area in which pooling is performed)

    strides : (int, tuple of two ints, or tuple of four ints)
        ** When given type is int **
            The output is subsampled by this factor in both width and
            height direction.

        ** When given type is tuple of two int **
            The output is subsapmled by ``strides[0]`` in height and
            ``striders[1]`` in width.

        Notes
            [Tensorflow only]

            When given type is tuple of four int, their order must be
            consistent with the input data format.

            **NHWC**: (batch, height, width, channel)

            **NCHW**: (batch, channel, height, width)

    padding : (str or int or tuple of two ints)
        Tensorflow
            Either 'SAME' or 'VALID'
        Theano
            Tuple of two ints or theano vector of ints of size 2.
            Also supports 'SAME' or 'VALID' for compatibility with
            Tensorflow backend.

        Notes
            When padding is 'SAME', due to the difference of backend
            implementation, Theano and Tensorflow cannot be numerically
            compatible.

    mode : str
        ``max`` or ``average``. Theano backend also supports ``sum``.
        Default: ``max``.

    name : str
        Used as base scope when building parameters and output

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys
    ``filter`` and ``bias`` in the same scope as layer build.
    """
    def __init__(
            self, kernel_height, kernel_width, strides, mode='max',
            padding='VALID', name='Pool2D'):
        super(Pool2D, self).__init__(
            kernel_height=kernel_height, kernel_width=kernel_width,
            strides=strides, mode=mode, padding=padding, name=name)
