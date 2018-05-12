import gym
import numpy as np
import tensorflow as tf
import random
import gym.spaces

from abc import ABC, abstractmethod


class Database:
    def __init__(self):
        self.database = {}

    def add(self, entry):
        for key, value in entry.items():
            value = np.array(value)
            if key not in self.database:
                self.database[key] = np.zeros([0] + list(value.shape))
            self.database[key] = np.concatenate([self.database[key], np.expand_dims(value, axis=0)], axis=0)

    def __getitem__(self, item):
        return self.database.__getitem__(item)


class Environment(ABC):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    @abstractmethod
    def step(self, actions):
        """
        :return: states, reward, done
        """
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class GymEnvironmentWrapper(Environment):
    def __init__(self, env_name):
        self.game = gym.make(env_name)
        self.done = False
        self.state = None

        super().__init__(self.game.action_space.n, self.game.observation_space.shape)

    def step(self, action):
        reward = 0
        state = self.state
        done = True

        if not self.done:
            state, reward, done, _ = self.game.step(action)
            self.state = state
            self.done = done

        return state, reward, done

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()

    def reset(self):
        self.done = False
        self.state = self.game.reset()
        return self.state


class MultipleEnvironment:
    def __init__(self, env_factory, num_instances=1):
        assert num_instances != 0
        self.num_instances = num_instances
        self.env_factory = env_factory
        self.environments = [self.env_factory() for _ in range(num_instances)]
        self.num_actions = self.environments[0].num_actions
        self.state_shape = self.environments[0].state_shape

        self.state = self.reset()

    def reset(self):
        states = []
        for environment in self.environments:
            states.append(environment.reset())
        self.state = states
        return states

    def step(self, actions):
        """
        :param actions: array of actions of shape [num_instances, num_actions]
        :return states: array of shape [num_instances, *state_shape]
                rewards: array of shape [num_instances]
               done: array of booleans with shape [num_instances]
        """
        assert (len(actions) == self.num_instances)
        states, rewards, dones = zip(*[self.environments[i].step(actions[i]) for i in range(self.num_instances)])
        return np.array(states), np.array(rewards), np.array(dones)

    def render(self, policy, max_steps):
        print("RENDERING")
        env = self.env_factory()
        state = env.reset()
        for _ in range(max_steps):
            env.render()
            state, _, done = env.step(np.squeeze(policy(state[np.newaxis])))
            if done:
                break
        env.close()

    def generate_trajectory(self, policy, max_steps, reset=True):
        """
        :param reset: Boolean whether to reset environments before generating trajectory
        :param max_steps: Steps to generate trajectory up to
        :param policy: A function from states [num_iterations, *state_shape] to actions [num_iterations, num_actions]
        :return states: An array of shape [num_instances, steps + 1, *state_shape],
                dones: An array of shape [num_instances, steps + 1]
                actions: An array of shape [num_instances, steps, num_actions]
                rewards: An array of shape [num_instances, steps]
                In each case, steps is the number of steps it takes to reach terminal or the max_steps
        """
        state = self.reset() if reset else self.state

        states, actions, rewards, dones = [], [], [], []
        dones = [np.zeros(self.num_instances, dtype=bool)]

        for i in range(max_steps):
            action = policy(state)
            states.append(state)

            state, reward, done = self.step(action)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            if all(done):
                break

        states.append(state)
        self.state = state
        return tuple(np.stack(values, axis=1) for values in (states, actions, rewards, dones))


class ActorCritic(ABC):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    @abstractmethod
    def actor(self, state):
        pass

    @abstractmethod
    def critic(self, state):
        pass


class BasicActorCritic(ActorCritic):
    def __init__(self, num_actions, state_shape, architecture=(32, 32), shared_architecture=()):
        super().__init__(num_actions, state_shape)
        self.architecture = architecture
        self.shared_architecture = shared_architecture

    def shared_layers(self, state):
        with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.shared_architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='shared%d' % i)
        return state

    def critic(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='critic%d' % i)
            state = tf.layers.dense(state, 1, activation=None, name='output')
        return tf.squeeze(state, axis=-1)

    def actor(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='layer%d' % i)
            probs = tf.layers.dense(state, units=self.num_actions, activation=tf.nn.softmax, name='output')
        return probs


class ConvActorCritic(ActorCritic):
    @staticmethod
    def shared_layers(state):
        with tf.variable_scope('ac_shared', reuse=tf.AUTO_REUSE):
            logits = tf.layers.conv2d(state, 16, kernel_size=7, strides=3, activation=tf.nn.leaky_relu, name='c1')
            logits = tf.layers.conv2d(logits, 16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, name='c2')
            logits = tf.layers.conv2d(logits, 16, kernel_size=3, activation=tf.nn.leaky_relu, name='c3')
            logits = tf.layers.conv2d(logits, 16, kernel_size=3, activation=tf.nn.leaky_relu, name='c4')
            logits = tf.layers.flatten(logits)
            logits = tf.layers.dense(logits, units=64, activation=tf.nn.leaky_relu, name='d1')
        return logits

    def critic(self, state):
        features = self.shared_layers(state)
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            return tf.layers.dense(features, units=1, activation=None, name='output')

    def actor(self, state):
        features = self.shared_layers(state)
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            return tf.layers.dense(features, units=self.num_actions, activation=tf.nn.softmax, name='output')


class Learner:
    def __init__(self, env_factory):
        tf.reset_default_graph()

        self.num_games = 4
        self.environments = MultipleEnvironment(env_factory, num_instances=self.num_games)

        self.num_actions = self.environments.num_actions
        self.state_shape = self.environments.state_shape

        self.discount_factor = 0.99
        self.segment_length = 10

        self.database = Database()
        self.prev_rewards = [0]

        self.state_input = tf.placeholder(tf.float32, [None, None, *self.state_shape], name='state_input')
        self.actions_input = tf.placeholder(tf.int32, [self.num_games, None], name='actions_input')
        self.non_terminals_input = tf.placeholder(tf.float32, [self.num_games, None], name='terminated_input')

        self.reward_mean = tf.placeholder(tf.float32, [], name='reward_mean')
        self.reward_std = tf.placeholder(tf.float32, [], name='reward_std')
        self.reward_input = tf.placeholder(tf.float32, [None], name='reward_input')
        self.discounted_reward_input = tf.placeholder(tf.float32, [self.num_games, None],
                                                      name='discounted_reward_input')

        # self.actor_critic = BasicActorCritic(self.game, shared_architecture=(32, 32), architecture=())
        self.actor_critic = BasicActorCritic(self.num_actions, self.state_shape,
                                             shared_architecture=[], architecture=[128, 128])

        actions_one_hot = tf.one_hot(self.actions_input, self.num_actions)
        self.predicted_rewards = self.reward(self.state_input, actions_one_hot)

        # -- CRITIC -- #
        self.state_value = self.actor_critic.critic(self.state_input)

        # normalized_reward = (self.predicted_rewards - self.reward_mean) / self.reward_std + 1
        advantage = self.discounted_reward_input - self.state_value

        # -- ACTOR -- #
        self.actor_probs = self.actor_critic.actor(self.state_input)
        action_probs = tf.reduce_sum(self.actor_probs * actions_one_hot, axis=-1)

        # -- TRAINING -- #
        self.entropy_bonus = tf.reduce_sum(0.01 * self.entropy(self.actor_probs))
        self.critic_loss = tf.reduce_sum(tf.square(advantage * self.non_terminals_input))
        self.actor_loss = -tf.reduce_sum(tf.log(action_probs) * tf.stop_gradient(advantage)
                                         * self.non_terminals_input)

        self.loss = self.actor_loss - self.entropy_bonus + self.critic_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # -- TRAIN REWARD -- #
        self.ratings_input = tf.placeholder(tf.float32, [None, 2])

        self.segment_inputs = {}

        predicted_rewards = []
        for s in ['s1', 's2']:
            segment = self.segment_inputs[s] = {}
            segment['states'] = tf.placeholder(tf.float32, [None, self.segment_length, *self.state_shape])
            segment['actions'] = tf.placeholder(tf.int32, [None, self.segment_length])
            segment['lengths'] = tf.placeholder(tf.int32, [None])

            mask = tf.sequence_mask(segment['lengths'], maxlen=self.segment_length, dtype=tf.float32)
            action_one_hot = tf.one_hot(segment['actions'], self.num_actions)
            predicted_rewards.append(tf.reduce_sum(self.reward(segment['states'], action_one_hot) * mask, axis=-1))

        reward_logits = tf.stack(predicted_rewards, axis=-1)

        self.reward_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ratings_input,
                                                                                     logits=reward_logits))
        self.reward_train_op = tf.train.AdamOptimizer(learning_rate=0.001, name='adam_reward') \
            .minimize(self.reward_loss)

        # -- INFERENCE -- #
        self.state_inf = tf.placeholder(tf.float32, shape=[*self.state_shape])
        state_inf = tf.expand_dims(self.state_inf, axis=0)
        self.output_inf = tf.squeeze(self.actor_critic.actor(state_inf))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def render(self):
        self.environments.render(self.policy, max_steps=1000)

    def policy(self, states):
        """
        :param states: Array of shape [num_instances, *state_shape]
        :return: Array of actions with shape [num_instances, num_actions]
        """
        action_probs = self.session.run(self.actor_probs, feed_dict={
            self.state_input: np.array(np.expand_dims(states, axis=1))
        })
        action_probs = np.squeeze(action_probs, axis=1)
        return np.stack([np.random.choice(self.num_actions, p=action_prob) for action_prob in action_probs])

    def save_model(self):
        save_path = self.saver.save(self.session, 'saves/model.ckpt')
        print("Model Saved in %s" % save_path)

    def load_model(self):
        self.saver.restore(self.session, 'saves/model.ckpt')
        print('Model Loaded')

    @staticmethod
    def entropy(probs):
        # return tf.reduce_sum(tf.log(1 - probs + 0.0001), axis=-1)
        return -tf.reduce_sum(probs * tf.log(probs + 0.0000001), axis=-1)

    @staticmethod
    def reward(states, actions):
        with tf.variable_scope('reward', reuse=tf.AUTO_REUSE):
            # state_action = tf.concat([states, actions], axis=-1)
            # logits = tf.layers.dense(state_action, 8, name='layer1', activation=tf.nn.relu)
            logits = tf.layers.dense(actions, 1, name='output')
        return tf.squeeze(logits)

    def train_policy(self, num_iterations=1):
        average_reward = 0
        average_loss = 0

        losses = np.zeros(3)

        for i in range(num_iterations):
            states, actions, rewards, terminals = self.environments.generate_trajectory(self.policy, max_steps=500)
            non_terminals = 1 - terminals

            # TODO: this depends on the parameters and should be included in back propagation
            # Calculate the values for the last time step for each run and use it if state is non-terminal
            bootstrap_values = self.session.run(self.state_value, feed_dict={
                self.state_input: states[:, [-1], :]
            }) * non_terminals[:, [-1]]

            discounted_rewards = np.copy(rewards)
            prev = np.squeeze(bootstrap_values)

            for j in range(1, states.shape[1]):
                discounted_rewards[:, -j] += prev * self.discount_factor
                prev = discounted_rewards[:, -j]

            fetches = [self.loss, self.predicted_rewards, self.train_op,
                       self.actor_loss, self.critic_loss, self.entropy_bonus,
                       self.state_value]

            results = self.session.run(fetches, feed_dict={
                self.state_input: states[:, :-1, :],
                self.non_terminals_input: non_terminals[:, :-1],
                self.actions_input: actions,
                self.discounted_reward_input: discounted_rewards
            })

            loss, predicted_rewards, _, actor_loss, critic_loss, entropy_bonus = results[:6]
            state_value = results[6:]  # For debug purposes

            self.prev_rewards = self.prev_rewards + list(predicted_rewards)
            self.prev_rewards = self.prev_rewards[-100:]
            losses += np.array([actor_loss, critic_loss, entropy_bonus])

            average_reward += np.sum(np.mean(rewards, axis=0))
            average_loss += loss

        print("\nActor Loss: %.3f\t Critic Loss %.3f\t Entropy Bonus %.3f" % tuple(losses / num_iterations))
        return average_reward / num_iterations, average_loss / num_iterations

    def train_reward(self):
        num_prev = 100

        feed_dict = {self.ratings_input: self.database['ratings'][-num_prev:]}
        for segment_key in ['s1', 's2']:
            for item_key in ['states', 'actions', 'lengths']:
                db_key = "%s_%s" % (segment_key, item_key)
                feed_dict[self.segment_inputs[segment_key][item_key]] = self.database[db_key][-num_prev:]

        loss, _ = self.session.run([self.reward_loss, self.reward_train_op], feed_dict=feed_dict)
        return loss

    def train_both(self, iterations=1, ratio=1):
        total_loss = 0
        total_reward = 0
        count_reward = 0
        for i in range(iterations):
            if i % ratio == 0:
                total_loss += self.train_reward()
                count_reward += 1
            total_reward += self.train_policy(1)[0]
        return total_reward / iterations, total_loss / count_reward

    def get_human_preference(self):
        seg1_states, seg1_actions, length1, total_reward1 = self.generate_segment()
        seg2_states, seg2_actions, length2, total_reward2 = self.generate_segment()

        prob = (np.sign(total_reward1 - total_reward2) + 1) / 2
        ratings = [prob, 1 - prob]

        self.database.add({
            'ratings': ratings,
            's1_states': seg1_states,
            's2_states': seg2_states,
            's1_actions': seg1_actions,
            's2_actions': seg2_actions,
            's1_lengths': length1,
            's2_lengths': length2
        })

    def generate_segment(self):
        states, actions, rewards = self.generate_trajectory()[:3]
        index = random.randint(1, len(states) - 1)
        seg_states = states[index:index + self.segment_length]
        seg_actions = actions[index: index + self.segment_length]

        length = len(seg_states)
        seg_states = self.pad_to_length(seg_states, self.segment_length)
        seg_actions = self.pad_to_length(seg_actions, self.segment_length)

        return seg_states, seg_actions, length, np.sum(rewards)

    @staticmethod
    def pad_to_length(tensor, length):
        dimensions = len(tensor.shape)

        pad_width = np.zeros([dimensions, 2], dtype=np.int32)
        pad_width[0, 1] = length - tensor.shape[0]
        return np.pad(tensor, pad_width, mode='constant')


def main():
    # env_name = 'Assault-ram-v0'
    # env_name = 'CartPole-v0'
    # env_name = 'MountainCar-v0'
    env_name = 'LunarLander-v2'
    # env_name = 'Pong-ram-v0'
    learner = Learner(lambda: GymEnvironmentWrapper(env_name))

    # for i in range(30):
    #     learner.get_human_preference()

    for i in range(500):
        learner.render()
        reward, loss = learner.train_policy(num_iterations=100)
        learner.save_model()
        print("Epoch %d\tReward: %.3f\tLoss: %.3f" % ((i + 1), reward, loss))

    # for i in range(30):
    #     learner.get_human_preference()
    #     reward, loss = learner.train_both(iterations=100, ratio=5)
    #     print("Reward: %.3f\t Reward Loss: %.3f" % (reward, loss))

    # states, actions, length, total_reward = learner.generate_segment()
    # predicted = learner.session.run([learner.predicted_rewards], feed_dict={
    #     learner.state_input: states,
    #     learner.actions_input: actions
    # })
    #
    # print(states)
    # print(predicted)
    # print(learner.prev_rewards)
    # print(np.mean(learner.prev_rewards))
    # print(np.std(learner.prev_rewards))
    #
    # print(learner.database.database['ratings'])
    # print(learner.database.database['s1_lengths'][:5])
    # print(learner.database.database['s2_lengths'][:5])

    return learner


if __name__ == '__main__':
    l = main()
