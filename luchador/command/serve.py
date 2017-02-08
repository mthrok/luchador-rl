"""Module to define ``luchador serve`` subcommand"""
from __future__ import absolute_import

import logging

from paste.translogger import TransLogger

import luchador.util
import luchador.env.remote
import luchador.nn as nn

_LG = logging.getLogger(__name__)


def _run_server(app, port):
    server = luchador.util.create_server(TransLogger(app), port=port)
    app.attr['server'] = server
    _LG.info('Starting server on port %d', port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        _LG.info('Server on port %d stopped.', port)


###############################################################################
def entry_point_env(args):
    """Entry point for ``luchador serve env`` command"""
    if args.environment is None:
        raise ValueError('Environment config is not given')
    env_config = luchador.util.load_config(args.environment)
    env = luchador.env.get_env(env_config['typename'])(**env_config['args'])
    app = luchador.env.remote.create_env_app(env)
    _run_server(app, args.port)


def entry_point_manager(args):
    """Entry point for ``luchador serve manager`` command"""
    app = luchador.env.remote.create_manager_app()
    _run_server(app, args.port)


def entry_point_param(args):
    """Entry point for ``luchador serve parameter``"""
    nn.make_model(args.model)
    session = nn.Session()
    session.initialize()
    app = luchador.nn.remote.create_parameter_server_app(session)
    _run_server(app, args.port)
