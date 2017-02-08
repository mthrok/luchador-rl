"""Define Parameter server which handle central roll in distributed training

"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

import flask

import luchador.nn
from luchador.util import serialize_numpy_array, deserialize_numpy_array
from ..util import make_io_node

_LG = logging.getLogger(__name__)


def _deserialize(dataset):
    """Deserialize NumPy ND Array dataset from JSON notation

    Parameters
    ----------
    data : dict
        data to deserialize. If a key is NumPy Array serialized with
        serialize_outcome function,  deserialize it. Otherwise, return data as
        it is.
    """
    dataset_ = OrderedDict()
    for key, array in dataset.items():
        input_ = luchador.nn.get_input(key)
        dataset_[input_] = deserialize_numpy_array(array)
    return dataset_


def _serialize(data):
    """Serialize NumPy ND Array dataset from JSON notation
    """
    return {
        key: serialize_numpy_array(value)
        for key, value in data.items()
    }


def _get_outputs(configs):
    return [
        make_io_node(config)
        for config in configs
    ]


def create_parameter_server_app(session):
    """Create Flask server for parameter processing

    See module documentation for the detail.


    Parameters
    ----------
    session : Session manager
        Session which holds operations

    """
    # pylint: disable=broad-except
    app = flask.Flask(__name__)
    attr = {
        'server': None
    }

    @app.route('/', methods=['POST', 'GET'])
    def _health():
        return 'Running\n'

    @app.route('/fetch', methods=['POST'])
    def _fetch_variable():
        """Fetch the value of Variables"""
        try:
            params = flask.request.get_json()
        except Exception:
            _LG.error('Failed to parse parameter')
            return 'Failed to parse parameter', 400

        if 'variables' not in params:
            return 'Missing parameter; "variables"', 400

        try:
            outputs = _get_outputs(params['variables'])
        except ValueError as error:
            _LG.error('Failed to parse: %s', str(error))
            return str(error), 400

        values = session.run(outputs=outputs)

        return_data = {
            key: serialize_numpy_array(value)
            for key, value in zip(params['variables'], values)
        }
        return flask.jsonify(**return_data)

    @app.route('/run', methods=['POST'])
    def _run():
        try:
            params = flask.request.get_json()
        except Exception:
            _LG.exception('Failed to parse parameter')
            return 'Failed to prase parmater', 400

        if 'name' not in params:
            return 'Missing paramter; "name"', 400

        inputs = _deserialize(params.get('inputs', {}))
        _LG.info(inputs)
        return ''

    app.attr = attr
    return app
