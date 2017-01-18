from __future__ import division
from __future__ import absolute_import

import os
import time
import logging

import cv2
import numpy as np
from rpi_rover.remote import client as rpi_client

import luchador.env

_LG = logging.getLogger(__name__)


def _deserialize_array(data, dtype, shape):
    return np.fromstring(data.decode('base64'), dtype=dtype).reshape(shape)


def _mean(list_):
    return sum(list_) / (len(list_) or 1)


def _convert_motor_conrtoll_to_reward(motor_controll):
    """Convert motor controll vector to reward

    Parameters
    ----------
    motor_controll : list of 2-tuple
        We assume
        - [(1, 1), (1, 1)]: The maximum forward drive
        - [(0, 0), (0, 0)]: Stop
        - [(-1, -1), (-1, -1)]: The maximum backword drive
        and reward to be simply the absolute scalar mean.
        (Thus encouraging rover to move forward/backword)

    Returns
    -------
    float
        Reward
    """
    vals = motor_controll[0] + motor_controll[1]
    sign = None
    for val in vals:
        if val == 0:
            continue
        if sign is None:
            sign = val > 0
        elif not sign == (val > 0):
            return 0
    return abs(_mean(vals))


def _convert_distances_to_terminal(distances, threshold):
    """Convert ultra sonice distance measurement to terminal flag

    Parameters
    ----------
    distances : list of float
        distance measurement from sensor
    threshold : float
        threshold for terminal flag. If any distance measurement is smaller
        than this threshold, terminal == True

    Returns
    -------
    Bool
        Terminal flag
    """
    return any(dist < threshold for dist in distances)


def _convert_status_to_outcome(status, distance_threshold, resize):
    """Convert the RPiRover status to environment Outcome"""
    if status:
        terminal = _convert_distances_to_terminal(
            status['distances'], distance_threshold)
        reward = -1 if terminal else _convert_motor_conrtoll_to_reward(
            status['motor_controll'])

        # _LG.debug('reward: %s -> %s', status['motor_controll'], reward)

        if resize:
            for img in ['gray_image', 'depth_image']:
                status[img] = cv2.resize(status[img], (resize[1], resize[0]))
        return luchador.env.Outcome(
            state={
                'gray_image': status['gray_image'],
                'depth_image': status['depth_image'],
            },
            reward=reward,
            terminal=terminal,
            info={
                'success': True,
                'timestamp': status['timestamp'],
            },
        )
    else:
        return luchador.env.Outcome(
            state=None,
            reward=None,
            terminal=None,
            info={
                'success': False,
                'timestamp': time.time(),
            }
        )


class RPiRover(luchador.env.BaseEnvironment):
    """Environment to interact with RPi Rover"""
    def __init__(
            self, host, port, distance_threshold, resize=None, mock=False):
        self.host = host
        self.port = port
        self.resize = tuple(resize) if resize else None
        self.distance_threshold = distance_threshold

        self.client = (
            rpi_client.ClientMock(host=host, port=port) if mock else
            rpi_client.Client(host=host, port=port)
        )
        self.client.setup()
        self.client.check_health()

    def _get_outcome(self):
        status = self.client.get_status()
        return _convert_status_to_outcome(
            status, self.distance_threshold, self.resize)

    def step(self, motor_controll):
        """Send command and fetch the current status

        Parameters
        ----------
        motor_controll : NumPy NDArray
            two or four values in range of [-1.0, 1.0]
        """
        param = motor_controll.reshape((-1, 2)).tolist()
        self.client.queue({'name': 'drive', 'param': param})
        return self._get_outcome()

    @property
    def n_actions(self):
        return 4

    def reset(self):
        _LG.debug('Resetting RPiRover')
        self.client.set_idle_time(0.15)
        for i in range(100):
            distances = self.client.get_distances()
            terminal = _convert_distances_to_terminal(
                distances, self.distance_threshold)
            _LG.debug('%d: %s %s', i, terminal, distances)

            if not terminal:
                break

            self.client.queue({
                'name': 'drive',
                'param': [(0.3, 0.3), (-0.3, -0.3)],
            })
            time.sleep(0.10)
        self.client.set_idle_time(0.3)
        self.client.stop_motor()
        return self._get_outcome()


###############################################################################
def _get_mock_data(dir_):
    ret = []
    dir_ = os.path.join(os.path.dirname(__file__), dir_)
    for sub_dir in os.listdir(dir_):
        path_ = os.path.join(dir_, sub_dir)
        for file_ in os.listdir(path_):
            ret.append(os.path.join(path_, file_))
    return ret


class RPiRoverMock(luchador.env.BaseEnvironment):
    def __init__(self, distance_threshold, resize):
        self.resize = resize
        self.distance_threshold = distance_threshold

        self._files = _get_mock_data('mock_data')
        self._data = None
        self._index = None

    @property
    def n_actions(self):
        return 4

    def _get_outcome(self):
        i = self._index
        terminal = _convert_distances_to_terminal(
            self._data['distances'][i, ...], self.distance_threshold)
        print self._data['distances'][i, ...], self.distance_threshold, terminal
        reward = -1 if terminal else _convert_motor_conrtoll_to_reward(
            self._data['motor_controll'][i, ...])
        ret = luchador.env.Outcome(
            state={
                'gray_image': np.copy(self._data['gray_image'][i, ...]),
                'depth_image': np.copy(self._data['gray_image'][i, ...]),
            },
            reward=reward,
            terminal=terminal,
            info={
                'success': True,
                'timestamp': self._data['timestamp'][i],
            },
        )
        self._index += 1
        return ret

    def reset(self):
        import h5py
        filepath = np.random.choice(self._files)
        self._data = h5py.File(filepath, mode='r')
        self._index = 0
        return self._get_outcome()

    def step(self, _):
        return self._get_outcome()
