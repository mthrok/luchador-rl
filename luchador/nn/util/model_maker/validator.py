"""Validate model configuration loaded from file"""
from __future__ import absolute_import

import os.path
import logging

from jsonschema import validate

from luchador.util import load_config

__all__ = ['validate_config']
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'schema')
_LG = logging.getLogger(__name__)
_BASIC_SCHEMA = {
    'type': 'object',
    'properties': {
        'typename': {'type': 'string'},
        'args': {'type': 'object'},
    },
    'required': ['typename']
}


def _load_schema(name):
    return load_config(os.path.join(_DATA_DIR, '{}.yml'.format(name)))


def validate_config(config):
    """Validate model config"""
    validate(config, _BASIC_SCHEMA)
    type_ = config['typename']
    try:
        schema = _load_schema(type_)
    except IOError:
        _LG.warn('No schema found for %s. Skipping validation.', type_)
        return
    validate(config.get('args', {}), schema)
