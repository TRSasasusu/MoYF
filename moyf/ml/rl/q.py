# coding: utf-8

import random
import numpy as np

class Q:
    RANDOM_RANGE = (0, 100)

    '''
    e.g.
    states_to_states_with_rewards = [
            [[1, 0], [2, 10]],
            [[0, 0], [3, 100]],
            [[0, 0]],
            [],
            ]
    Every state has its next states and rewards.
    '''
    def __init__(self, states_to_states_with_rewards):
        self.states_to_states_with_rewards = states_to_states_with_rewards
        self.action_value_function = [[random.randint(*Q.RANDOM_RANGE) for i in to_states_with_rewards] for to_states_with_rewards in self.states_to_states_with_rewards]

    def epsilon_greedy(self, epsilon, learning_rate, discount_rate, limit=100):
        agent_state = 0
        for i in range(100):
            if len(self.states_to_states_with_rewards[agent_state]) == 0:
                break

            if random.random() < epsilon:
                action_index = random.randint(0, len(self.states_to_states_with_rewards[agent_state]) - 1)
            else:
                action_index = np.argmax(self.action_value_function[agent_state])

            next_agent_state, reward = self.states_to_states_with_rewards[agent_state][action_index]

            if len(self.action_value_function[next_agent_state]) == 0:
                max_next_action_value_function = self.action_value_function[agent_state][action_index]
                #max_next_action_value_function = 0
            else:
                max_next_action_value_function = max(self.action_value_function[next_agent_state])

            self.action_value_function[agent_state][action_index] += learning_rate * (
                    reward + discount_rate * max_next_action_value_function - self.action_value_function[agent_state][action_index]
                    )

            agent_state = next_agent_state

    def print_action_value_function(self, place, is_line_broken=True):
        print('[', end='')
        for row in self.action_value_function:
            print('[', end='')
            for value in row:
                print('{0:.{1}f}, '.format(value, place), end='')
            print('], ', end='')
        print(']', end='')

        if is_line_broken:
            print('')
