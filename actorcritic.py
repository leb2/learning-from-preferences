from abc import ABC, abstractmethod
import tensorflow as tf


class ActorCritic(ABC):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    @abstractmethod
    def actor(self, state):
        """
        Tensorflow operation that maps states to actions

        :param state: State is tensor of shape [num_instances, num_steps, *state_shape]
        :return: A tensor of shape [num_instances, num_steps, num_actions]
        """
        pass

    @abstractmethod
    def critic(self, state):
        """
        Tensorflow operation that maps states to values

        :param state: State is tensor of shape [num_instances, num_steps, *state_shape]
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


class BasicActorCritic(ActorCritic):
    def __init__(self, num_actions, state_shape, architecture=(32, 32), shared_architecture=()):
        super().__init__(num_actions, state_shape)
        self.architecture = architecture
        self.shared_architecture = shared_architecture

    # TODO: Reshape
    def shared_layers(self, state):
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


class ConvActorCritic(ActorCritic):
    def shared_layers(self, state):
        batch_shape = tf.shape(state)[:-len(self.state_shape)]
        logits = tf.reshape(state, [-1, *self.state_shape])

        with tf.variable_scope('ac_shared', reuse=tf.AUTO_REUSE):
            logits = tf.layers.conv2d(logits, 16, kernel_size=7, strides=3, activation=tf.nn.leaky_relu, name='c1')
            logits = tf.layers.conv2d(logits, 16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, name='c2')
            logits = tf.layers.conv2d(logits, 16, kernel_size=3, activation=tf.nn.leaky_relu, name='c3')
            logits = tf.layers.conv2d(logits, 16, kernel_size=3, activation=tf.nn.leaky_relu, name='c4')
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

