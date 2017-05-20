"""Test Q-Learning module"""
from __future__ import division
from __future__ import absolute_import

import os.path

import luchador.nn as nn


def _get_data_dir():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, 'data')


_DATA_DIR = _get_data_dir()


def get_model_config(model_name, **parameters):
    """Load model configurations from library or file

    Parameters
    ----------
    model_name : str
        Model name or path to YAML file

    parameters
        Parameter for model config

    Returns
    -------
    JSON-compatible object
        Model configuration.
    """
    if not model_name.endswith('.yml'):
        model_name = '{}.yml'.format(model_name)
    if os.path.isfile(model_name):
        file_path = os.path.join(_DATA_DIR, model_name)
    elif os.path.isfile(os.path.join(_DATA_DIR, model_name)):
        file_path = os.path.join(_DATA_DIR, model_name)
    return nn.util.get_model_config(file_path, **parameters)
