import numpy as np
import tensorflow as tf
import random

from environments import MultipleEnvironment, GymEnvironmentWrapper, MusicEnvironment, PongEnvironment
from actorcritic import BasicActorCritic, ConvActorCritic
from util import Util


class Database:
    def __init__(self, max_size=5000):
        self.database = {}
        self.length = 0
        self.max_size = max_size
        self.batch_index = 0

    def add(self, entry):
        self.length += 1

        for key, value in entry.items():
            value = np.array(value)
            if key not in self.database:
                self.database[key] = np.zeros([0] + list(value.shape))
            self.database[key] = np.concatenate([self.database[key], np.expand_dims(value, axis=0)], axis=0)

        if self.length > self.max_size:
            for key, value in self.database.items():
                self.database[key] = value[-(self.max_size - 100):]
            self.length = self.max_size

    def get_next_batch(self, batch_size):
        batch_size = min(batch_size, self.length)
        # rand_index = np.random.randint(0, self.length)
        batch = {
            key: data.take(range(self.batch_index, self.batch_index + batch_size), mode='wrap', axis=0)
            for key, data in self.database.items()
        }
        self.batch_index += batch_size
        return batch

    def __getitem__(self, item):
        return self.database.__getitem__(item)


class Learner:
    def __init__(self, env_factory, use_predicted_rewards=True):
        tf.reset_default_graph()

        self.use_predicted_rewards = use_predicted_rewards
        self.num_games = 1
        self.environments = MultipleEnvironment(env_factory, num_instances=self.num_games)

        self.num_actions = self.environments.num_actions
        self.state_shape = self.environments.state_shape

        self.discount_factor = 0.99
        self.segment_length = 10

        self.database = Database()
        self.validation_database = Database()

        self.saved_trajectories = []
        self.max_saved_trajectories = 100

        self.prev_rewards = [0]

        self.in_train_mode = tf.placeholder(tf.bool, [])
        self.state_input = tf.placeholder(tf.float32, [None, None, *self.state_shape], name='state_input')
        self.actions_input = tf.placeholder(tf.int32, [self.num_games, None], name='actions_input')
        self.non_terminals_input = tf.placeholder(tf.float32, [self.num_games, None], name='terminated_input')
        self.discounted_reward_input = tf.placeholder(tf.float32, [self.num_games, None],
                                                      name='discounted_reward_input')

        self.actor_critic = ConvActorCritic(self.num_actions, self.state_shape)
        # self.actor_critic = BasicActorCritic(self.num_actions, self.state_shape,
        #                                      shared_architecture=[], architecture=[128, 128],
        #                                      reward_architecture=[64, 32], dropout_prob=0.5, num_ensemble=3)

        actions_one_hot = tf.one_hot(self.actions_input, self.num_actions)
        self.state_value = self.actor_critic.critic(self.state_input)

        advantage = self.discounted_reward_input - self.state_value
        self.actor_probs = self.actor_critic.actor(self.state_input)
        action_probs = tf.reduce_sum(self.actor_probs * actions_one_hot, axis=-1)

        self.predicted_rewards = self.actor_critic.reward(self.state_input, actions_one_hot)

        self.entropy_bonus = tf.reduce_sum(0.01 * self.entropy(self.actor_probs))
        self.critic_loss = tf.reduce_sum(tf.square(advantage * self.non_terminals_input))
        self.actor_loss = -tf.reduce_sum(tf.log(action_probs) * tf.stop_gradient(advantage)
                                         * self.non_terminals_input)
        self.loss = self.actor_loss - self.entropy_bonus + self.critic_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
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
            predicted_rewards.append(tf.reduce_sum(self.actor_critic.reward(
                segment['states'], action_one_hot, use_dropout=True) * mask, axis=-1))

        reward_logits = tf.stack(predicted_rewards, axis=-1)
        self.reward_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.ratings_input, logits=reward_logits))
        self.reward_train_op = tf.train.AdamOptimizer(learning_rate=0.001, name='adam_reward') \
            .minimize(self.reward_loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.use_predicted_rewards:
            self.populate_saved_trajectories(num_trajectories=50)

    def render(self):
        self.environments.render(self.policy)

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
        return -tf.reduce_sum(probs * tf.log(probs + 1e-10), axis=-1)

    def predict_rewards(self, states, actions):
        raw_rewards = self.session.run(self.predicted_rewards, feed_dict={
            self.state_input: states,
            self.actions_input: actions
        })
        return self.normalize_rewards(raw_rewards)

    def train_policy(self):
        trajectories = self.environments.generate_trajectory(self.policy)
        states, actions, true_rewards, terminals = trajectories

        self.saved_trajectories += [trajectories]
        self.saved_trajectories = self.saved_trajectories[-self.max_saved_trajectories:]

        rewards = true_rewards
        if self.use_predicted_rewards:
            rewards = self.predict_rewards(states[:, :-1], actions)

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

        # TODO: Don't consider predicted rewards at terminals
        self.prev_rewards = (self.prev_rewards + list(predicted_rewards[0]))[-1000:]

        reward = np.sum(np.mean(true_rewards, axis=0))
        return reward

    def train_reward(self, validation=False):
        database = self.database if not validation else self.validation_database
        batch_size = 256 if not validation else 5000

        batch = database.get_next_batch(batch_size)
        feed_dict = {self.ratings_input: batch['ratings']}

        for segment_key in ['s1', 's2']:
            for item_key in ['states', 'actions', 'lengths']:
                db_key = "%s_%s" % (segment_key, item_key)
                feed_dict[self.segment_inputs[segment_key][item_key]] = batch[db_key]

        fetches = [self.reward_loss]
        feed_dict[self.in_train_mode] = not validation

        if not validation:
            fetches.append(self.reward_train_op)
        loss = self.session.run(fetches, feed_dict=feed_dict)[0]
        return loss

    def normalize_rewards(self, rewards):
        """
        Normalizes an array of rewards using the history of rewards.

        :param rewards: An array of shape [num_instances, num_steps]
        :return: An array with the same shape of
        """
        mean = np.mean(np.array(self.prev_rewards))
        std = np.std(np.array(self.prev_rewards))
        std = std if std != 0 else 1
        return (rewards - mean) / std

    def save_human_preferences(self, use_validation_db=False):
        seg1_states, seg1_actions, length1, total_reward1 = self.generate_segment()
        seg2_states, seg2_actions, length2, total_reward2 = self.generate_segment()

        # The segment that 'wins' gets 1 probability and the other segments gets 0. If there is a tie,
        # both segments get 0.5 probability
        prob = (np.sign(total_reward1 - total_reward2) + 1) / 2
        ratings = [prob, 1 - prob]

        database = self.database if not use_validation_db else self.validation_database
        database.add({
            'ratings': ratings,
            's1_states': seg1_states,
            's2_states': seg2_states,
            's1_actions': seg1_actions,
            's2_actions': seg2_actions,
            's1_lengths': length1,
            's2_lengths': length2
        })

    def populate_saved_trajectories(self, num_trajectories):
        for i in range(num_trajectories):
            print("Populating trajectory %d" % i)
            trajectory = self.environments.generate_trajectory(self.policy)
            self.saved_trajectories += [trajectory]

    def generate_segment(self):
        # TODO: This is incorrect for non zero terminals
        rand_index = np.random.choice(len(self.saved_trajectories))
        states, actions, rewards, _ = self.saved_trajectories[rand_index]

        # Only care about first run in list of trajectories
        states = states[0, :-1]
        actions = actions[0]
        rewards = rewards[0]

        index = random.randint(1, actions.shape[0] - 1)
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


def normal_actor_critic():
    # learner = Learner(MusicEnvironment,  use_predicted_rewards=False)
    # learner = Learner(lambda: GymEnvironmentWrapper('LunarLander-v2'), use_predicted_rewards=False)
    learner = Learner(lambda: GymEnvironmentWrapper('BeamRider-v0', max_steps=1000), use_predicted_rewards=False)
    # learner.load_model()

    for i in range(1000):
        reward = Util.train_multiple(learner.train_policy, num_iterations=10)
        print("Epoch %d\treward %.3f" % ((i + 1), reward))
        if i % 5 == 0:
            learner.render()
        learner.save_model()


def main():
    # learner = Learner(MusicEnvironment)
    learner = Learner(lambda: GymEnvironmentWrapper('LunarLander-v2'))

    for i in range(1000):
        print("Human preference %d" % (i + 1))
        use_validation_db = i % 5 == 0
        learner.save_human_preferences(use_validation_db=use_validation_db)

    # Train reward function until validation loss goes up (early stopping)
    min_val_loss = float('inf')
    for _ in range(5000):
        Util.train_multiple(learner.train_reward, num_iterations=1)
        validation_loss = Util.train_multiple(learner.train_reward, num_iterations=1, validation=True)
        min_val_loss = min(validation_loss, min_val_loss)
        print(validation_loss)
        if validation_loss > min_val_loss + 0.025:
            break

    for i in range(500):
        print("\n\n")
        if i % 20 == 0:
            learner.render()

        for j in range(1):
            learner.save_human_preferences(use_validation_db=False)
            learner.save_human_preferences(use_validation_db=True)

        reward = Util.train_multiple(learner.train_policy, num_iterations=10)
        loss = Util.train_multiple(learner.train_reward, num_iterations=1)
        validation_loss = Util.train_multiple(learner.train_reward, num_iterations=1, validation=True)
        learner.render()
        learner.save_model()

        print("Epoch %d \t Reward: %.3f \t Reward Loss: %.3f \t Val Loss: %.3f"
              % ((i + 1), reward, loss, validation_loss))
        print("Currently there are %d items in the database" % learner.database.length)


if __name__ == '__main__':
    # main()
    normal_actor_critic()
