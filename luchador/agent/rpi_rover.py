from __future__ import division
from __future__ import absolute_import

import logging

import numpy as np

import luchador.agent
from luchador.util import StoreMixin
from luchador import nn

from .misc import OUNoise
from .recorder import TransitionRecorder
from .rl.ddpg import ContinuousActorCriticNetwork

_LG = logging.getLogger(__name__)


def _flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


class RPiRoverControl(StoreMixin, luchador.agent.BaseAgent):
    def __init__(
            self,
            recorder_config,
            model_config,
            q_network_config,
            training_config,
            noise_config,
            image_type='depth_image',
    ):
        self._store_args(
            recorder_config=recorder_config,
            model_config=model_config,
            q_network_config=q_network_config,
            training_config=training_config,
            noise_config=noise_config,
            image_type=image_type,
        )

        self._network = None
        self._recorder = None
        self._noise = None

        self._n_actions = None
        self._n_train = 0
        self._n_obs = 0

    def init(self, env):
        self._n_actions = env.n_actions

        cfg = self.args['model_config']
        fmt = luchador.get_nn_conv_format()

        c, h, w = cfg['stacks'], cfg['height'], cfg['width']
        input_shape = (
            '[null, {}, {}, {}]'.format(c, h, w) if fmt == 'NCHW' else
            '[null, {}, {}, {}]'.format(h, w, c)
        )

        model_def = nn.get_model_config(
            cfg['model_file'], n_motors=self._n_actions,
            input_shape=input_shape)

        self._network = ContinuousActorCriticNetwork(
            **self.args['q_network_config'])
        self._network.build(model_def)

        self._recorder = TransitionRecorder(**self.args['recorder_config'])
        self._noise = OUNoise(
            shape=(1, self._n_actions), **self.args['noise_config']
        )

    def reset(self, state):
        self._recorder.reset({
            'state': state[self.args['image_type']]
        })

    def learn(self, state0, action, reward, state1, terminal, info):
        self._recorder.record({
            'action': action,
            'state': state1[self.args['image_type']],
            'reward': reward,
            'terminal': terminal})
        self._n_obs += 1

        cfg, n_obs = self.args['training_config'], self._n_obs
        if cfg['start'] < 0 or n_obs < cfg['start']:
            return

        if n_obs == cfg['start']:
            _LG.info('Starting Network Training')

        if n_obs % cfg['sync'] == 0:
            _LG.debug('Syncing Network')
            self._network.sync()

        if n_obs % cfg['update'] == 0:
            self._train_network(cfg['n_samples'])
            self._n_train += 1

    def _train_network(self, n_samples):
        # _LG.debug('Training Network')
        samples = self._recorder.sample(n_samples)
        q_value = self._network.train(
            state0=samples['state'][0],
            action=samples['action'],
            reward=samples['reward'],
            state1=samples['state'][1],
            terminal=samples['terminal'],
        )
        # TODO: Check network output magnitude and rewrad magniture
        _LG.debug('Q value: %s', q_value)

    def act(self):
        action = self._noise.sample()
        if self._recorder.is_ready():
            stacks = self._recorder.get_last_stack()
            input_ = stacks['state'][None, ...]
            action_ = self._network.get_action(input_)
            _LG.debug('action_:  %s', action_)
            action += action_
            # self._actions_debug.append(action_[0])
        _action = np.copy(action)
        action[action > 1.0] = 1.0
        action[action < -1.0] = -1.0
        _LG.debug('action: %s -> %s', _action, action)
        return action

    def perform_post_episode_task(self, stats):
        self._recorder.truncate()
        '''
        import matplotlib.pyplot as plt
        _actions = np.asarray(self._actions_debug)
        print _actions.shape
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.scatter(_actions[:, 0], _actions[:, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect(1)
        ax = fig.add_subplot(2, 1, 2)
        ax.scatter(_actions[:, 2], _actions[:, 3])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect(1)
        plt.show()
        self._actions_debug = self._actions_debug[-100:]
        '''
