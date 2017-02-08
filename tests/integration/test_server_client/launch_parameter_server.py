"""Launch parameter server"""
from __future__ import absolute_import

import logging
import argparse

import luchador
import luchador.util
from luchador.util import initialize_logger
import luchador.nn as nn
import luchador.nn.remote
from luchador.agent.rl.q_learning import DeepQLearning

_LG = logging.getLogger(__name__)


def _build_model():
    c, h, w, fmt = 4, 84, 84, luchador.get_nn_conv_format()
    input_shape = [None, c, h, w] if fmt == 'NCHW' else [None, h, w, c]

    model_def = nn.get_model_config(
        'vanilla_dqn', n_actions=4, input_shape=input_shape)

    dql = DeepQLearning(
        q_learning_config={
            'discount_rate': 0.99,
            'min_reward': -1.0,
            'max_reward': 1.0,
        },
        optimizer_config={
            'typename': 'RMSProp',
            'args': {
                'decay': 0.95,
                'epsilon': 1e-6,
                'learning_rate': 2.5e-4,
            },
        },
    )
    dql.build(model_def, None)
    return dql


def _run_server(app, port):
    server = luchador.util.create_server(app, port=port)
    _LG.info('Starting server on port %d', port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        _LG.info('Server on port %d stopped.', port)


def _parser_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port', default=5000, type=int,
        help='Port to run Parameter Server'
    )
    parser.add_argument(
        '--debug', action='store_true',
    )
    return parser.parse_args()


def _initialize_logger(debug):
    message_format = (
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
        if debug else '%(asctime)s: %(levelname)5s: %(message)s'
    )
    level = logging.DEBUG if debug else logging.INFO
    initialize_logger(
        name='luchador', message_format=message_format, level=level)


def _main():
    args = _parser_command_line_args()
    _initialize_logger(args.debug)

    dql = _build_model()
    app = nn.remote.create_parameter_server_app(dql.session)
    _run_server(app, args.port)


if __name__ == '__main__':
    _main()
