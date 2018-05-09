import gym
import numpy as np
import tensorflow as tf
import random
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


class ActorCritic(ABC):
    def __init__(self, game):
        self.num_actions = game.action_space.n
        self.state_shape = game.observation_space.shape

    @abstractmethod
    def actor(self, state):
        pass

    @abstractmethod
    def critic(self, state):
        pass


class BasicActorCritic(ActorCritic):
    def __init__(self, game, architecture=(32, 32), shared_architecture=()):
        super().__init__(game)
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
        return tf.squeeze(state)

    def actor(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='layer%d' % i)
            probs = tf.layers.dense(state, units=self.num_actions, activation=tf.nn.softmax, name='output')
        return probs


class ConvActorCritic(ActorCritic):
    def __init__(self, game):
        super().__init__(game)

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
    def __init__(self, game):
        tf.reset_default_graph()
        self.batch_size = 30
        self.game = game
        self.gamma = 1

        self.num_actions = game.action_space.n
        self.state_shape = game.observation_space.shape

        self.segment_length = 10

        self.database = Database()
        self.prev_rewards = [0]

        self.state_input = tf.placeholder(tf.float32, shape=[None, *self.state_shape], name='state_input')
        self.actions_input = tf.placeholder(shape=[None], dtype=tf.int32, name='actions_input')
        self.reward_mean = tf.placeholder(tf.float32, [], name='reward_mean')
        self.reward_std = tf.placeholder(tf.float32, [], name='reward_std')
        self.reward_input = tf.placeholder(tf.float32, [None], name='reward_input')
        self.discounted_reward_input = tf.placeholder(tf.float32, [None], name='discounted_reward_input')

        # self.actor_critic = BasicActorCritic(self.game, shared_architecture=(32, 32), architecture=())
        self.actor_critic = BasicActorCritic(self.game, shared_architecture=[], architecture=[64, 64])

        actions_one_hot = tf.one_hot(self.actions_input, self.num_actions)
        self.predicted_rewards = self.reward(self.state_input, actions_one_hot)

        # -- CRITIC -- #
        state_value = self.actor_critic.critic(self.state_input)

        # normalized_reward = (self.predicted_rewards - self.reward_mean) / self.reward_std + 1
        normalized_reward = self.reward_input

        # One-step Advantage
        next_state_value = self.gamma * tf.concat([state_value[1:], [0]], axis=0)
        td_advantage = normalized_reward + next_state_value - state_value

        self.next_state_value = next_state_value

        # Monte Carlo Advantage
        # cumulative = tf.cumsum(normalized_reward, reverse=True)
        # advantage = cumulative - tf.squeeze(state_value)
        mc_advantage = self.discounted_reward_input - state_value

        self.state_value = state_value
        self.td_advantage = td_advantage
        self.mc_advantage = mc_advantage

        # -- ACTOR -- #
        output = self.actor_critic.actor(self.state_input)
        action_probs = tf.reduce_sum(output * actions_one_hot, axis=1)

        # -- TRAINING -- #
        self.entropy_bonus = tf.reduce_mean(0.01 * self.entropy(output))

        # self.critic_loss = tf.reduce_mean(tf.square(advantage))
        self.critic_loss = tf.reduce_mean(tf.square(td_advantage))
        self.actor_loss = -tf.reduce_mean(tf.log(action_probs) * tf.stop_gradient(td_advantage))

        self.loss = self.actor_loss - self.entropy_bonus + self.critic_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)

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

    def generate_trajectory(self, render=False, greedy=False):
        state = self.game.reset()
        rewards, actions, states = [], [], []
        total_reward = 0
        for step in range(3000):
            if render:
                self.game.render()
            probabilities = self.session.run(self.output_inf, feed_dict={
                self.state_inf: state
            })

            if greedy:
                action = np.argmax(probabilities)
            else:
                action = np.random.choice(self.num_actions, p=probabilities)
            # action = np.random.choice(self.num_actions)

            actions.append(action)
            states.append(state)
            state, reward, done, _ = self.game.step(action)

            rewards.append(reward)
            total_reward += reward

            if done:
                break
        if render:
            self.game.close()

        return np.array(states), np.array(actions), np.array(rewards), total_reward

    def train_policy(self, num_iterations=1):
        average = 0
        average_loss = 0

        losses = np.zeros(3)

        for i in range(num_iterations):
            # sys.stdout.write("game: %d of %d \r" % (i, num_iterations))

            states, actions, rewards, total_reward = self.generate_trajectory()
            discounted_rewards = [0.0]

            for j, reward in enumerate(rewards):
                discounted_rewards[0] += reward * self.gamma ** j

            for j in range(1, len(rewards)):
                discounted_rewards.append(float(discounted_rewards[j - 1] - rewards[j - 1]) / self.gamma)

            fetches = [self.loss, self.predicted_rewards, self.train_op,
                       self.actor_loss, self.critic_loss, self.entropy_bonus,
                       self.td_advantage, self.mc_advantage, self.state_value, self.next_state_value]

            results = self.session.run(fetches, feed_dict={
                self.state_input: np.array(states),
                self.actions_input: np.array(actions),
                self.reward_input: np.array(rewards),
                self.discounted_reward_input: np.array(discounted_rewards),
                self.reward_mean: np.mean(self.prev_rewards),
                self.reward_std: np.std(self.prev_rewards) + 0.00001
            })
            loss, predicted_rewards, _, actor_loss, critic_loss, entropy_bonus = results[:6]
            td_advantage, mc_advantage, state_values, next_state_value = results[6:]

            self.prev_rewards = self.prev_rewards + list(predicted_rewards)
            self.prev_rewards = self.prev_rewards[-100:]
            losses += np.array([actor_loss, critic_loss, entropy_bonus])

            average_loss += loss
            average += total_reward

        print("Actor Loss: %.3f\t Critic Loss %.3f\t Entropy Bonus %.3f" % tuple(losses / num_iterations))
        return average / num_iterations, average_loss / num_iterations

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
    # environment = gym.make('Assault-ram-v0')
    # environment = gym.make('CartPole-v0')
    # environment = gym.make('MountainCar-v0')
    environment = gym.make('Pong-ram-v0')
    # learner = Learner(gym.make('LunarLander-v2'))

    learner = Learner(environment)
    # learner.load_model()

    # for i in range(30):
    #     learner.get_human_preference()

    for i in range(1000):
        reward, loss = learner.train_policy(num_iterations=30)
        print("Epoch %d\tReward: %.3f\tLoss: %.3f" % ((i + 1), reward, loss))
        if i % 10 == 0:
            learner.save_model()
    learner.generate_trajectory(True)

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
