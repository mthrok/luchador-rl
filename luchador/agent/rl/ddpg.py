from __future__ import division
from __future__ import absolute_import

import luchador
from luchador import nn


class ContinuousActorCriticNetwork(object):
    def __init__(self, discount_rate, optimizer_configs):
        self.discount_rate = discount_rate
        self.optimizer_configs = optimizer_configs

        self._session = None
        self._var = None
        self._ops = None
        self._models = None
        self._optimizers = None

    def build(self, model_def):
        # Construct image preprocessor, actor and critic for t and t+1
        with nn.variable_scope('t_0'):
            actor0, critic0 = nn.make_model(model_def)
            state0, action0 = actor0.input, actor0.output
            q_value0 = critic0.output
        with nn.variable_scope('t_1'):
            actor1, critic1 = nn.make_model(model_def)
            state1, action1 = actor1.input, actor1.output
            q_value1 = critic1.output

        # Build cost function for Q value
        reward = nn.Input(shape=(None, 1), name='reward')
        terminal = nn.Input(shape=(None, 1), name='terminal')
        sse2 = nn.cost.SSE2(max_delta=1.0, min_delta=-1.0)
        target_q = reward + (1 - terminal) * q_value1 * self.discount_rate
        q_error = sse2(target_q, q_value0)

        # Build optimization operations
        # TODO:
        #     For actor update: check Deterministic policy gradient algorithms
        #     http://jmlr.org/proceedings/papers/v32/silver14.pdf
        #     https://arxiv.org/pdf/1509.02971v5.pdf
        #     https://github.com/stevenpjg/ddpg-aigym

        opt_cfg = self.optimizer_configs['actor']
        actor_optimizer = nn.get_optimizer(
            opt_cfg['typename'])(**opt_cfg.get('args', {}))
        opt_cfg = self.optimizer_configs['critic']
        critic_optimizer = nn.get_optimizer(
            opt_cfg['typename'])(**opt_cfg.get('args', {}))

        var_actor0 = actor0.get_parameter_variables()
        var_critic0 = critic0.get_parameter_variables()

        with nn.variable_scope('optimize_actor'):
            update_actor_op = actor_optimizer.minimize(
                -q_value0.mean(), wrt=var_actor0)
        with nn.variable_scope('optimize_critic'):
            update_critic_op = critic_optimizer.minimize(
                q_error, wrt=var_critic0)

        # Build sync operations
        var_actor1 = actor1.get_parameter_variables()
        var_critic1 = critic1.get_parameter_variables()

        sync_ops = [
            nn.build_sync_op(
                target_vars=var_actor1, source_vars=var_actor0, tau=0.9),
            nn.build_sync_op(
                target_vars=var_critic1, source_vars=var_critic0, tau=0.9),
        ]

        session = nn.Session()
        session.initialize()
        session.run(updates=sync_ops)

        self._session = session

        self._var = {
            'state0': state0,
            'state1': state1,
            'action0': action0,
            'action1': action1,
            'reward': reward,
            'terminal': terminal,
            'q_value0': q_value0,
            'q_value1': q_value1,
            'q_error': q_error,
        }

        self._ops = {
            'sync': sync_ops,
            'update_actor': update_actor_op,
            'update_critic': update_critic_op,
        }

        self._models = {
            'actor0': actor0,
            'actor1': actor1,
            'critic0': critic0,
            'critic1': critic1,
        }

        self._optimizers = {
            'actor': actor_optimizer,
            'critic': critic_optimizer,
        }

    def get_action(self, state):
        if luchador.get_nn_conv_format() == 'NHWC':
            state = state.transpose((0, 2, 3, 1))
        return self._session.run(
            outputs=self._var['action0'],
            givens={self._var['state0']: state},
            name='get_action',
        )

    def sync(self):
        self._session.run(
            updates=self._ops['sync'],
            name='sync',
        )

    def train(self, state0, action, reward, state1, terminal):
        if luchador.get_nn_conv_format() == 'NHWC':
            state0 = state0.transpose((0, 2, 3, 1))
            state1 = state1.transpose((0, 2, 3, 1))
        # Update actor
        self._session.run(
            updates=[
                self._ops['update_actor'],
            ],
            inputs={
                self._var['state0']: state0,
            },
            name='update_actor',
        )
        # Update critic wit double DQN
        action1 = self._session.run(
            outputs=self._var['action0'],
            inputs={self._var['state0']: state1},
        )
        self._session.run(
            updates=[
                self._ops['update_critic'],
            ],
            inputs={
                self._var['state0']: state0,
                self._var['reward']: reward,
                self._var['terminal']: terminal,
                self._var['action1']: action1,
                self._var['state1']: state1,
            },
            givens={
                self._var['action0']: action,
            },
            name='update_critic',
        )
        # Get value for debuging
        values = self._session.run(
            outputs=[
                self._var['q_value0'],
                self._var['q_value1'],
                self._var['q_error'],
            ],
            inputs={
                self._var['state0']: state0,
                self._var['state1']: state1,
                self._var['reward']: reward,
                self._var['terminal']: terminal,
            },
        )
        return [val.mean() for val in values]
