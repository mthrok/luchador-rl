from __future__ import division
from __future__ import absolute_import

import os
import time
import logging
from collections import defaultdict

import h5py
import numpy as np
from rpi_rover.remote import client as rpi_client

import luchador.env
from .rover_controll import _convert_distances_to_terminal


_LG = logging.getLogger(__name__)


def _save(filepath, records):
    _LG.info('Saving: %s', filepath)
    file_ = h5py.File(filepath, 'w')
    for key, value in records.items():
        file_.create_dataset(key, data=np.array(value))
    file_.close()


class Recorder(object):
    def __init__(self, directory):
        self.directory = os.path.join(directory, str(time.time()))

        self.records = defaultdict(list)
        self.episode = 0

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def record(self, **status):
        for key, value in status.items():
            self.records[key].append(value)

    def reset(self):
        if self.records:
            filename = 'episode_{:04}.h5'.format(self.episode)
            filepath = os.path.join(self.directory, filename)
            _save(filepath, self.records)
            self.episode += 1
            self.records = defaultdict(list)


class RPiRoverRecorder(luchador.env.BaseEnvironment):
    def __init__(
            self, host, port, distance_threshold=30,
            mock=False, directory='record'):
        self.host = host
        self.port = port

        self.distance_threshold = distance_threshold

        self._step = 0

        self._client = (
            rpi_client.ClientMock(host=host, port=port) if mock else
            rpi_client.Client(host=host, port=port)
        )
        self._client.setup()
        self._client.check_health()
        self._recorder = Recorder(directory)
        self._reset_dir = 0

    @property
    def n_actions(self):
        return 0

    def step(self, _):
        """Move rover forward and fetch state"""
        if self._step == 0:
            _LG.info('Starting step')
        self._step += 1

        resp = self._client.queue(
            {'name': 'drive', 'param': [[0.3, 0.3], [0.3, 0.3]]}
        )
        if resp['status_code'] == 412:
            _LG.warning('Queue is full.')
            time.sleep(0.3)

        status = self._client.get_status()
        self._recorder.record(
            distances=status['distances'],
            motor_controll=status['motor_controll'],
            gray_image=status['gray_image'],
            depth_image_x=status['depth_image']['x'],
            depth_image_y=status['depth_image']['y'],
            depth_image_z=status['depth_image']['z'],
            timestamp=['timestamp'],
        )
        terminal = _convert_distances_to_terminal(
            status['distances'][:1], self.distance_threshold)
        reward = -1 if terminal else 0

        if terminal:
            _LG.info('Stop')
            _LG.info('Saving Data')
            self._recorder.reset()

        return luchador.env.Outcome(
            state=None, reward=reward, terminal=terminal)

    def reset(self):
        _LG.info('Resetting RPiRover')
        self._step = 0

        t0 = time.time()
        _LG.info('    Backing up')
        for i in range(30):
            distances = self._client.get_distances()
            _LG.debug('%d: %s', i, distances)

            if distances[1] < 30:
                break

            self._client.queue({
                'name': 'drive',
                'param': [(-0.3, -0.3), (-0.3, -0.3)],
            })
            time.sleep(0.15)

        if (self._reset_dir % 4) // 2:
            param = [(0.33, 0.33), (-0.33, -0.33)]
        else:
            param = [(-0.33, -0.33), (0.33, 0.33)]
        self._reset_dir += 1

        t0 = time.time()
        _LG.info('    Rotating')
        while time.time() - t0 < 1.0:
            distances = self._client.get_distances()

            self._client.queue({
                'name': 'drive',
                'param': param,
            })
            time.sleep(0.15)
        self._client.stop_motor()
        _LG.info('Resetting Done')
        return luchador.env.Outcome(state=None, reward=None, terminal=False)
