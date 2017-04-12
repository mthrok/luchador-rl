from __future__ import division
from __future__ import absolute_import

import os.path
import logging
from collections import OrderedDict

import numpy as np

import luchador
import luchador.nn as nn
from luchador.util import StoreMixin, pprint_dict
from .base import BaseAgent
from .misc import EGreedy
from .recorder import PrioritizedQueue
from .rl import DeepQLearning, DoubleDeepQLearning

_LG = logging.getLogger(__name__)


def _build_optimize_op(optimizer, loss, params, clip_grads):
    grads_and_vars = nn.ops.compute_gradient(loss=loss, wrt=params)
    # Remove untrainable variables
    grads_and_vars = [g_v for g_v in grads_and_vars if g_v[0] is not None]
    if clip_grads:
        grads_and_vars = nn.ops.clip_grads_by_norm(
            grads_and_vars, **clip_grads)
    return optimizer.apply_gradients(grads_and_vars)


class Predictor(StoreMixin):
    def __init__(self, optimizer_config, **_):
        super(Predictor, self).__init__()
        self._store_args(
            optimizer_config=optimizer_config)

        self.vars = None
        self.models = None
        self.ops = None
        self.optimizer = None
        self.session = None

    def build(self, model_def, session, input_shape, batch_size):
        encoder0 = nn.make_model(model_def['encoder'])
        encoder1 = nn.make_model(model_def['encoder'])
        decoder = nn.make_model(model_def['decoder'])

        state0 = nn.Input(shape=input_shape, name='state0')
        state1 = nn.Input(shape=input_shape, name='state1')
        action = nn.Input(shape=(batch_size, 1, 1, 1), name='action')
        weight = nn.Input(shape=(batch_size,), name='sample_weight_predictor')

        with nn.variable_scope('mapped'):
            mapped0 = encoder0(state0)

        with nn.variable_scope('displacement'):
            # mapped1 = encoder0(state1)
            displacement = nn.ops.multiply(encoder1(state0), action)

        mapped1 = mapped0 + displacement

        with nn.variable_scope('decode0'):
            state0_pred = decoder(mapped0)

        with nn.variable_scope('decode1'):
            state1_pred = decoder(mapped1)

        with nn.variable_scope('recon_error'):
            recon_error0 = nn.cost.SSE(elementwise=True, name='recon_error0')(
                target=state0, prediction=state0_pred)
            recon_error1 = nn.cost.SSE(elementwise=True, name='recon_error1')(
                target=state1, prediction=state1_pred)
            recon_error = nn.ops.reduce_sum(
                recon_error0 + recon_error1, axis=[1, 2, 3])

        with nn.variable_scope('loss'):
            weighted_error = recon_error * weight
            loss = nn.ops.reduce_mean(weighted_error)

        params = (
            encoder0.get_parameters_to_train() +
            encoder1.get_parameters_to_train() +
            decoder.get_parameters_to_train()
        )

        cfg = self.args['optimizer_config']
        optimizer = nn.fetch_optimizer(cfg['typename'])(**cfg['args'])
        optimize_op = _build_optimize_op(
            optimizer=optimizer, loss=loss, params=params,
            clip_grads=self.args.get('clip_grads')
        )

        self.session = session

        self.models = {
            'encoder_0': encoder0,
            'encoder_1': encoder1,
            'decoder': decoder,
        }
        self.vars = {
            'state_0': state0,
            'state_1': state1,
            'mapped_0': mapped0,
            'mapped_1': mapped1,
            'displacement': displacement,
            'state_0_recon': state0_pred,
            'state_1_recon': state1_pred,
            'action': action,
            'recon_error': recon_error,
            'weight': weight,
        }
        self.ops = {
            'optimize': optimize_op,
        }
        self.optimizer = optimizer

    def train(self, state0, action, state1, weight=None):
        if weight is None:
            weight = np.ones((action.size, ), dtype=self.vars['weight'].dtype)

        return self._train(state0, action, state1, weight)

    def _train(self, state_0, action, state_1, weight):
        updates = (
            self.models['encoder_0'].get_update_operations() +
            self.models['encoder_1'].get_update_operations() +
            self.models['decoder'].get_update_operations() +
            [self.ops['optimize']]
        )
        return self.session.run(
            outputs=[
                self.vars['recon_error'],
            ],
            inputs={
                self.vars['state_0']: state_0,
                self.vars['action']: action,
                self.vars['state_1']: state_1,
                self.vars['weight']: weight,
            },
            updates=updates,
            name='minibatch_training',
        )

    def get_parameters_to_serialize(self):
        """Fetch network parameters and optimizer parameters for saving"""
        params = (
            self.models['encoder_0'].get_parameters_to_serialize() +
            self.models['encoder_1'].get_parameters_to_serialize() +
            self.models['decoder'].get_parameters_to_serialize() +
            self.optimizer.get_parameters_to_serialize()
        )
        params_val = self.session.run(outputs=params, name='save_params')
        return OrderedDict([
            (var.name, val) for var, val in zip(params, params_val)
        ])

    ###########################################################################
    def get_parameters_to_summarize(self):
        """Fetch parameters of each layer"""
        params = (
            self.models['encoder_0'].get_parameters_to_serialize() +
            self.models['encoder_1'].get_parameters_to_serialize() +
            self.models['decoder'].get_parameters_to_serialize()
        )
        params_vals = self.session.run(outputs=params, name='model_0_params')
        return {
            v.name.replace('/', '_', 1): val
            for v, val in zip(params, params_vals)
        }

    def get_layer_outputs(self, state, action):
        """Fetch outputs from each layer

        Parameters
        ----------
        state : NumPy ND Array
            Input to model0 (pre-transition model)
        """
        outputs = (
            self.models['encoder_0'].get_output_tensors() +
            self.models['encoder_1'].get_output_tensors() +
            self.models['decoder'].get_output_tensors()
        )
        output_vals = self.session.run(
            outputs=outputs,
            inputs={self.vars['state_0']: state, self.vars['action']: action},
            name='model_0_outputs'
        )
        return {
            v.name.replace('/', '_', 1): val
            for v, val in zip(outputs, output_vals)
        }

    ###########################################################################
    def predict(self, state_0, action):
        """Given state and action, predict the next state"""
        return self.session.run(
            outputs=[self.vars['state_0_recon'], self.vars['state_1_recon']],
            inputs={
                self.vars['state_0']: state_0,
                self.vars['action']: action,
            },
            name='prediction',
        )


def _gen_model_def(config, n_actions, batch_size=None):
    fmt, w = luchador.get_nn_conv_format(), config['input_width']
    h, c = config['input_height'], config['input_channel']
    model = config['model_file']
    shape = [batch_size, h, w, c] if fmt == 'NHWC' else [batch_size, c, h, w]
    return nn.get_model_config(model, n_actions=n_actions, input_shape=shape)


def _get_q_network(config):
    if config['typename'] == 'DeepQLearning':
        dqn = DeepQLearning
    elif config['typename'] == 'DoubleDeepQLearning':
        dqn = DoubleDeepQLearning
    return dqn(**config['args'])


def _initialize_summary_writer(config, session):
    writer = nn.SummaryWriter(**config)

    sess = nn.get_session()
    if sess.graph:
        writer.add_graph(sess.graph)

    return writer


def _transpose(state):
    return state.transpose((0, 2, 3, 1))


class PredictionAgent(StoreMixin, BaseAgent):
    def __init__(
            self,
            record_config,
            recorder_config,
            actor_config,
            q_network_config,
            predictor_config,
            saver_config,
            save_config,
            summary_writer_config,
            summary_config,
            action_config,
            training_config,
    ):
        self._store_args(
            record_config=record_config, recorder_config=recorder_config,
            actor_config=actor_config, q_network_config=q_network_config,
            predictor_config=predictor_config, saver_config=saver_config,
            save_config=save_config, summary_config=summary_config,
            summary_writer_config=summary_writer_config,
            action_config=action_config, training_config=training_config,
        )
        self._n_obs = 0
        self._n_train = 0
        self._n_actions = None
        self._ready = False

        self._recorder = None
        self._saver = None
        self._session = None
        self._ql = None
        self._pred = None
        self._eg = None
        self._summary_writer = None
        self._summary_values = {
            'total_errors': [],
            'recon_errors': [],
            'latent_errors': [],
            'mapping_errors': [],
            'rewards': [],
            'steps': [],
            'episode': 0,
        }

    ###########################################################################
    # Methods for initialization
    def init(self, env):
        self._n_actions = env.n_actions
        self._recorder = PrioritizedQueue(**self.args['recorder_config'])

        self._session = nn.get_session()
        self._init_actor(n_actions=env.n_actions)
        self._init_predictor(n_actions=env.n_actions)
        self._init_parameter()
        self._ql.sync_network()

        self._eg = EGreedy(**self.args['action_config'])
        self._saver = nn.Saver(**self.args['saver_config'])
        self._summary_writer = _initialize_summary_writer(
            config=self.args['summary_writer_config'],
            session=self._session,
        )
        self._summarize_layer_params()

    def _init_actor(self, n_actions):
        config = self.args['actor_config']
        model_def = _gen_model_def(config, n_actions)
        _LG.info('Creating actor network\n%s', pprint_dict(model_def))
        self._ql = _get_q_network(config=self.args['q_network_config'])
        self._ql.build(model_def, session=self._session)

    def _init_predictor(self, n_actions):
        config = self.args['predictor_config']
        self._pred = Predictor(**config)
        model_def = nn.get_model_config(config['model_file'])
        _LG.info('Creating predictor network\n%s', pprint_dict(model_def))
        if luchador.get_nn_conv_format() == 'NHWC':
            input_shape = [32, 84, 84, 1]
        else:
            input_shape = [32, 1, 84, 84]
        self._pred.build(model_def, session=self._session, batch_size=32, input_shape=input_shape)

    def _init_parameter(self):
        _LG.info('Initializing parameters')
        self._session.initialize()
        if self.args['actor_config']['initial_parameter']:
            self._session.load_from_file(
                self.args['actor_config']['initial_parameter'])
        if self.args['predictor_config']['initial_parameter']:
            self._session.load_from_file(
                self.args['predictor_config']['initial_parameter'])

    ###########################################################################
    # Methods for `reset`
    def reset(self, _):
        self._ready = False

    ###########################################################################
    # Methods for `act`
    def act(self):
        if not self._ready or self._eg.act_random():
            return np.random.randint(self._n_actions)

        q_val = self._predict_q()
        return np.argmax(q_val)

    def _predict_q(self):
        # _LG.debug('Predicting Q value from NN')
        state = self._recorder.get_last_record()['state1'][None, ...]
        if luchador.get_nn_conv_format() == 'NHWC':
            state = _transpose(state)
        return self._ql.predict_action_value(state)[0]

    ###########################################################################
    # Methods for `learn`
    def learn(self, state0, action, reward, state1, terminal, info=None):
        self._n_obs += 1
        self._record(state0, action, reward, state1, terminal)
        self._train()

    def _record(self, state0, action, reward, state1, terminal):
        """Stack states and push them to recorder, then sort memory"""
        self._recorder.push(1, {
            'state0': state0, 'action': action, 'reward': reward,
            'state1': state1, 'terminal': terminal})
        self._ready = True

        cfg = self.args['record_config']
        sort_freq = cfg['sort_frequency']
        if sort_freq > 0 and self._n_obs % sort_freq == 0:
            _LG.info('Sorting Memory')
            self._recorder.sort()
            _LG.debug('Sorting Complete')

        save_freq = cfg.get('save_frequency', 0)
        if save_freq > 0 and self._n_obs % save_freq == 0:
            _LG.info('Saiving Replay Memory')
            self._save_record()

    def _save_record(self):
        save_dir = self.args['record_config']['save_dir']
        for id_, record in self._recorder.id2record.items():
            path = os.path.join(save_dir, str(id_))
            np.savez_compressed(path, **record)

    # -------------------------------------------------------------------------
    # Training
    def _train(self):
        """Schedule training"""
        cfg = self.args['training_config']
        if cfg['train_start'] < 0 or self._n_obs < cfg['train_start']:
            return

        if self._n_obs == cfg['train_start']:
            _LG.info('Starting predictor training')

        if self._n_obs % cfg['train_frequency'] == 0:
            errors = self._train_network()
            self._n_train += 1
            # self._summary_values['total_errors'].append(errors[0])
            self._summary_values['recon_errors'].append(errors[0])
            # self._summary_values['latent_errors'].append(errors[2])
            # self._summary_values['mapping_errors'].append(errors[3])
            self._save_and_summarize()

    def _sample(self):
        """Sample transition from recorder and build training batch"""
        data = self._recorder.sample()
        records = data['records']
        state0 = np.asarray([r['state0'] for r in records])
        state1 = np.asarray([r['state1'] for r in records])
        action = np.asarray([r['action'] for r in records], dtype='float32')
        weights, indices = data['weights'], data['indices']
        samples = {
            'state0': state0, 'state1': state1,
            'action': action, 'weight': weights,
        }
        return samples, indices

    def _train_network(self):
        """Train network"""
        samples, indices = self._sample()
        samples['state0'] = samples['state0'][:, -1:, :, :] / 255
        samples['state1'] = samples['state1'][:, -1:, :, :] / 255
        samples['state0'] = samples['state0'].astype(np.float32)
        samples['state1'] = samples['state1'].astype(np.float32)
        samples['action'] = samples['action'].reshape((-1, 1, 1, 1))
        if luchador.get_nn_conv_format() == 'NHWC':
            samples['state0'] = _transpose(samples['state0'])
            samples['state1'] = _transpose(samples['state1'])
        errors = self._pred.train(**samples)
        self._recorder.update(indices, np.abs(errors[0]))
        return errors

    # -------------------------------------------------------------------------
    # Save and summarize
    def _save_and_summarize(self):
        """Save model parameter and summarize occasionally"""
        interval = self.args['save_config']['interval']
        if interval > 0 and self._n_train % interval == 0:
            _LG.info('Saving parameters')
            self._save_parameters()

        interval = self.args['summary_config']['interval']
        if interval > 0 and self._n_train % interval == 0:
            _LG.info('Summarizing Network')
            self._summarize_layer_params()
            self._summarize_layer_outputs()
            self._summarize_history()

    def _save_parameters(self):
        """Save trained parameters to file"""
        data = self._pred.get_parameters_to_serialize()
        self._saver.save(data, global_step=self._n_train)

    def _summarize_layer_params(self):
        """Summarize layer parameter statistic"""
        dataset = self._pred.get_parameters_to_summarize()
        self._summary_writer.summarize(
            global_step=self._n_train, summary_type='histogram', dataset=dataset)

    def _summarize_layer_outputs(self):
        """Summarize layer output"""
        samples, _ = self._sample()
        state0 = samples['state0'][:, -1:, :, :] / 255
        state0 = state0.astype(np.float32)
        action = samples['action'].reshape((-1, 1, 1, 1))
        if luchador.get_nn_conv_format() == 'NHWC':
            state0 = _transpose(state0)
        dataset = self._pred.get_layer_outputs(state0, action)
        self._summary_writer.summarize(
            global_step=self._n_train, summary_type='histogram', dataset=dataset)

    def _summarize_history(self):
        """Summarize training history"""
        steps = self._summary_values['steps']
        total_errors = self._summary_values['total_errors']
        recon_errors = self._summary_values['recon_errors']
        mapping_errors = self._summary_values['mapping_errors']
        latent_errors = self._summary_values['latent_errors']
        rewards = self._summary_values['rewards']
        episode = self._summary_values['episode']
        self._summary_writer.summarize(
            summary_type='histogram',
            global_step=self._n_train,
            dataset={
                'Training/TotalError': total_errors,
                'Training/ReconError': recon_errors,
                'Training/MappingError': mapping_errors,
                'Training/LatentError': latent_errors,
                'Training/Reward': rewards,
                'Training/Steps': steps,
            },
        )
        self._summary_writer.summarize(
            summary_type='scalar',
            global_step=episode, dataset={'Trainings': self._n_train}
        )
        if rewards:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'Reward': rewards}
            )
        if total_errors:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'TotalError': total_errors}
            )
        if mapping_errors:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'MappingError': mapping_errors}
            )
        if recon_errors:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'ReconError': recon_errors}
            )
        if latent_errors:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'LatentError': latent_errors}
            )
        if steps:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'Steps': steps}
            )
        self._summary_values['total_errors'] = []
        self._summary_values['recon_errors'] = []
        self._summary_values['latent_errors'] = []
        self._summary_values['mapping_errors'] = []
        self._summary_values['rewards'] = []
        self._summary_values['steps'] = []

    ###########################################################################
    # Methods for post_episode_action
    def perform_post_episode_task(self, stats):
        self._summary_values['rewards'].append(stats['rewards'])
        self._summary_values['steps'].append(stats['steps'])
        self._summary_values['episode'] = stats['episode']


class PredictionAgentTest(PredictionAgent):
    def learn(self, state0, action, reward, state1, terminal, info=None):
        self._n_obs += 1
        self._record(state0, action, reward, state1, terminal)

    def perform_post_episode_task(self, stats):
        if len(self._recorder) < 32:
            return

        if np.random.rand() < 0.95:
            return

        import matplotlib.pyplot as plt
        samples, _ = self._sample()
        samples['state0'] = samples['state0'][:, -1:, :, :] / 255
        samples['state0'] = samples['state0'].astype(np.float32)
        samples['state1'] = samples['state1'][:, -1:, :, :] / 255
        samples['state1'] = samples['state1'].astype(np.float32)
        samples['action'] = samples['action'].reshape((-1, 1, 1, 1))

        if luchador.get_nn_conv_format() == 'NHWC':
            samples['state0'] = _transpose(samples['state0'])
            samples['state1'] = _transpose(samples['state1'])

        state0_pred, state1_pred = self._pred.predict(
            samples['state0'], samples['action'])

        for batch in range(32):
            fig = plt.figure()
            fig.suptitle('Action: {}'.format(samples['action'][0, 0, 0, 0]))

            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(samples['state0'][batch][0])
            ax.set_title('State 0')

            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(state0_pred[batch][0])
            ax.set_title('Reconstructed State 0')

            ax = fig.add_subplot(2, 2, 3)
            ax.imshow(samples['state1'][batch][0])
            ax.set_title('State 1')

            ax = fig.add_subplot(2, 2, 4)
            ax.imshow(state1_pred[batch][0])
            ax.set_title('Reconstructed State 1')
        plt.show()
