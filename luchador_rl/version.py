"""Define version number"""
from __future__ import absolute_import

import pkg_resources
__version__ = pkg_resources.require('luchador_rl')[0].version
