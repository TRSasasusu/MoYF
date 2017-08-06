# coding: utf-8

from functools import reduce
import numpy as np
from moyf.ml.common import diff

class REINFORCE:
    RANDOM_RANGE = 200
    DIFF_H = 0.0001

    @staticmethod
    def policy_softmax(expected_values):
        max_value = max(expected_values)
        values = np.exp(np.array(expected_values) - max_value)
        return values / np.sum(values)

    @staticmethod
    def decide_action(policy_values):
        return argmax(policy_values)

    def __init__(self, expected_values_callback, decide_action_callback, calc_state_reward_callback, theta_dim, policy_callback):
        self.expected_values_callback = expected_values_callback
        self.decide_action_callback = decide_action_callback
        self.calc_state_reward_callback = calc_state_reward_callback
        self.theta = np.random.random_sample(theta_dim) * REINFORCE.RANDOM_RANGE - REINFORCE.RANDOM_RANGE * 0.5
        self.policy_callback = policy_callback

    def update(self, start_state, learning_rate, limit=100, episode_num=10):
        rewards = []
        time_lengths = []
        states = []
        actions = []
        for m in range(episode_num):
            rewards.append([])
            actions.append([])
            states.append([])
            state = start_state
            states[m].append(state)
            for t in range(limit):
                expected_values = self.policy_callback(state, self.theta)
                policies = self.policy_callback(expected_values)
                action = decide_action_callback(policies)
                state, reward, is_end = calc_state_reward_callback(action, state, self.theta)
                rewards[m].append(reward)
                actions[m].append(action)
                states[m].append(state)
                if is_end:
                    break
            time_lengths.append(t + 1)

        mean_rewards = reduce(lambda x, y: x + y, [sum(partial_rewards) for partial_rewards in rewards]) / sum(time_lengths)
        gradient = np.zeros(shape=theta_dim)
        for m in range(episode_num):
            for t in range(time_lengths[m]):
                policy_gradient = np.zeros(shape=theta_dim)
                for i, theta_row in enumerate(theta):
                    for j, theta_row_elem in enumerate(theta_row):
                        tmp_theta = self.theta.copy()
                        tmp_theta[i][j] -= DIFF_H
                        left_expected_values = self.policy_callback(states[m][t], tmp_theta)
                        tmp_theta[i][j] += DIFF_H * 2
                        right_expected_values = self.policy_callback(states[m][t], tmp_theta)

                        left_policy = self.policy_callback(left_expected_values)[actions[m][t]]
                        right_policy = self.policy_callback(right_expected_values)[actions[m][t]]

                        policy_gradient[i][j] = (np.log(right_policy) - np.log(left_policy)) / (2 * DIFF_H)

                gradient += (rewards[m][t] - mean_rewards) * policy_gradient

        self.theta += learning_rate * gradient
