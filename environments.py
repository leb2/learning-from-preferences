import gym
import gym.spaces
from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    @abstractmethod
    def step(self, action):
        """
        :param action: an integer representing action index
        :return: states, reward, done
        """
        pass

    @abstractmethod
    def render(self):
        """
        Function called at each time step when rendering is enabled
        :return:
        """
        pass

    @abstractmethod
    def close(self):
        """
        Function called when rendering completes
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to initial state (for episodic environments)
        """
        pass


class MusicEnvironment(Environment):
    def __init__(self):
        self.words = list('-tyuiasdfgh')
        self.word_indices = {i: word for i, word in enumerate(self.words)}
        self.num_prev = 5

        num_actions = len(self.words)
        state_shape = [self.num_prev * len(self.words)]
        super().__init__(num_actions, state_shape)

        self.state = self.start_state()
        self.render_buffer = []

    def reward(self, state, action):
        prev_notes = self.state_to_notes(state)
        prev_note = prev_notes[-1]
        action_note = self.word_indices[action]

        return float(action_note == '-' and prev_note != '-')

    def step(self, action):
        action = int(action)
        reward = self.reward(self.state, action)

        state = np.pad(self.state, ((0, 1), (0, 0)), mode='constant')[1:, :]
        state[-1, action] = 1
        self.state = state

        return state.flatten(), reward, False

    def render(self):
        action = self.state_to_notes(self.state)[-1]
        self.render_buffer.append(action)

    def close(self):
        print(''.join(self.render_buffer))
        self.render_buffer = []

    def reset(self):
        self.state = self.start_state()
        return self.state.flatten()

    def state_to_notes(self, state):
        indices = []
        for time_step in state:
            index = list(time_step).index(1)
            indices.append(self.word_indices[index])
        return ''.join(indices)

    def start_state(self):
        state = np.zeros([self.num_prev, len(self.words)])
        state[:, 0] = 1
        return state


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

