from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class ActorCritic(ABC):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    @abstractmethod
    def actor(self, states):
        """
        Tensorflow operation that maps states to actions

        :param states: A tensor of shape [num_instances, num_steps, *state_shape]
        :return: A tensor of shape [num_instances, num_steps, num_actions]
        """
        pass

    @abstractmethod
    def critic(self, states):
        """
        Tensorflow operation that maps states to values

        :param states: A tensor of shape [num_instances, num_steps, *state_shape]
        :return: A tensor of shape [num_instances, num_steps]
        """
        pass

    @abstractmethod
    def reward(self, states, actions):
        """
        Tensorflow operation that maps states to values

        :param states: A tensor of shape [num_instances, num_steps, *state_shape]
        :param actions: A one hot tensor of indices of shape [num_instances, num_steps, num_actions]
        :return: A tensor of shape [num_instances, num_steps]
        """
        pass


class LSTMActorCritic(ActorCritic):
    def __init__(self, num_actions, state_shape, name='lstm_ac'):
        self.name = name
        super().__init__(num_actions, state_shape)
        self.cell = tf.contrib.BasicLSTMCell()

    def actor(self, state, initial_state=None):
        with tf.variable_scope('%s/actor' % self.name):
            tf.nn.dynamic_rnn(self.cell)

    def critic(self, state, initial_state=None):
        with tf.variable_scope('%s/critic' % self.name):
            tf.nn.dynamic_rnn(self.cell)

    def reward(self, states, actions):
        raise NotImplementedError


class BasicActorCritic(ActorCritic):
    def __init__(self, num_actions, state_shape, architecture=(32, 32), shared_architecture=(),
                 reward_architecture=None, dropout_prob=0, num_ensemble=1):

        super().__init__(num_actions, state_shape)
        self.architecture = architecture
        self.shared_architecture = shared_architecture
        self.reward_architecture = architecture if reward_architecture is None else reward_architecture
        self.dropout_prob = dropout_prob
        self.num_ensemble = num_ensemble

    def shared_layers(self, state):
        state = tf.reshape(state, tf.concat([tf.shape(state)[:2], [np.prod(self.state_shape)]], axis=0))
        with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.shared_architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='shared%d' % i)
        return state

    def actor(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='layer%d' % i)
            probs = tf.layers.dense(state, units=self.num_actions, activation=tf.nn.softmax, name='output')
        return probs

    def critic(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='critic%d' % i)
            state = tf.layers.dense(state, 1, activation=None, name='output')
        return tf.squeeze(state, axis=-1)

    def reward(self, states, actions, use_dropout=None):
        states = self.shared_layers(states)
        with tf.variable_scope('reward', reuse=tf.AUTO_REUSE):
            logits = tf.concat([states, actions], axis=-1)
            ensemble_total = 0

            for e in range(self.num_ensemble):
                for i, layer_size in enumerate(self.reward_architecture):
                    logits = tf.layers.dense(logits, layer_size, activation=tf.nn.tanh,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='critic%d_e%d' % (i, e))
                    if use_dropout is not None:
                        logits = tf.layers.dropout(logits, rate=self.dropout_prob, training=use_dropout)
                logits = tf.layers.dense(logits, 1, activation=None,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         name='output_e%d' % e)
                ensemble_total += logits
        return tf.squeeze(ensemble_total, axis=-1) / self.num_ensemble


class ConvActorCritic(ActorCritic):
    def __init__(self, num_actions, state_shape, num_ensemble=1, dropout_prob=0):
        self.reward_architecture = (32,)
        self.num_ensemble = num_ensemble
        self.dropout_prob = dropout_prob

        self.filters = [16, 16, 16, 16]
        self.kernel_sizes = [7, 5, 3, 3]
        self.strides = [3, 2, 1, 1]
        self.architecture = list(zip(range(len(self.filters)), self.filters, self.kernel_sizes, self.strides))

        super().__init__(num_actions, state_shape)

    def shared_layers(self, state):
        batch_shape = tf.shape(state)[:-len(self.state_shape)]
        logits = tf.reshape(state, [-1, *self.state_shape])

        with tf.variable_scope('ac_shared', reuse=tf.AUTO_REUSE):
            for i, filters, kernel_size, strides in self.architecture:
                logits = tf.layers.conv2d(logits, filters, kernel_size, strides, activation=tf.nn.leaky_relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          name='conv%d' % i)
            logits = tf.layers.flatten(logits)
            logits = tf.layers.dense(logits, units=64, activation=tf.nn.leaky_relu, name='d1')

        logits = tf.reshape(logits, tf.concat([batch_shape, [64]], axis=0))
        return logits

    def actor(self, state):
        features = self.shared_layers(state)
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            return tf.layers.dense(features, units=self.num_actions, activation=tf.nn.softmax, name='output')

    def critic(self, state):
        features = self.shared_layers(state)
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            features = tf.layers.dense(features, units=1, activation=None, name='output')
        return tf.squeeze(features, axis=-1)

    def reward(self, states, actions, use_dropout=None):
        states = self.shared_layers(states)
        with tf.variable_scope('reward', reuse=tf.AUTO_REUSE):
            logits = tf.concat([states, actions], axis=-1)
            ensemble_total = 0

            for e in range(self.num_ensemble):
                for i, layer_size in enumerate(self.reward_architecture):
                    logits = tf.layers.dense(logits, layer_size, activation=tf.nn.tanh,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='critic%d_e%d' % (i, e))
                    if use_dropout is not None:
                        logits = tf.layers.dropout(logits, rate=self.dropout_prob, training=use_dropout)
                logits = tf.layers.dense(logits, 1, activation=None,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         name='output_e%d' % e)
                ensemble_total += logits
        return tf.squeeze(ensemble_total, axis=-1) / self.num_ensemble

