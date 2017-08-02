# coding: utf-8

from moyf.ml.rl import q

def main():
    states_to_states_with_rewards = [
            [[1, 1], [2, 0]],
            [[0, -1], [3, 1]],
            [[0, -100], [3, 5]],
            [],
            ]
    ql = q.Q(states_to_states_with_rewards)
    for i in range(1000):
        print('{0}: '.format(i), end='')
        ql.print_action_value_function(2)
        ql.epsilon_greedy(0.05, 0.1, 0.9)
    print('result: ', end='')
    ql.print_action_value_function(2)

if __name__ == '__main__':
    main()
