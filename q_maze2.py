# coding: utf-8

import sys
import pygame
from pygame.locals import Rect, QUIT
from moyf.ml.rl import q

WIDTH = 600
HEIGHT = 600
START_STATE = 12
GOAL_STATE = 5

def main():
    def view(agent_state):
        pygame.display.set_caption("Q-Learning trial: {0}".format(trial_number))
        screen.fill((0, 0, 0))

        for i, position in enumerate(positions):
            left_top = (position[0] * WIDTH / 6, position[1] * HEIGHT / 6)
            if i == START_STATE or i == GOAL_STATE:
                color = (255, 255, 255)
            else:
                color = (128, 128, 128)
            pygame.draw.rect(screen, color, Rect(left_top[0], left_top[1], WIDTH / 6, HEIGHT / 6))

            if i == agent_state:
                pygame.draw.ellipse(screen, (0, 0, 255), Rect(left_top[0], left_top[1], WIDTH / 6, HEIGHT / 6))

        pygame.display.update()
        pygame.time.wait(100)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        

    states_to_states_with_rewards = [
            [[1, 0]], # 0
            [[0, 0], [2, 0]], # 1
            [[1, 0], [3, 0], [6, 0]], # 2
            [[2, 0], [4, 0]], # 3
            [[3, 0], [7, 0]], # 4
            [], # 5
            [[2, 0], [10, 0]], # 6
            [[4, 0], [8, 0]], # 7
            [[7, 0], [9, 0], [11, 0]], # 8
            [[8, 0], [5, 100]], # 9
            [[6, 0], [13, 0]], # 10
            [[8, 0], [15, 0]], # 11
            [[13, 0]], # 12
            [[12, 0], [10, 0], [14, 0]], # 13
            [[13, 0], [17, 0]], # 14
            [[11, 0], [16, 0]], # 15
            [[15, 0], [18, 0]], # 16
            [[14, 0]], # 17
            [[16, 0]], # 18
            ]
    positions = [
            (0, 5), # 0
            (1, 5), # 1
            (1, 4), # 2
            (2, 4), # 3
            (3, 4), # 4
            (5, 4), # 5
            (1, 3), # 6
            (3, 3), # 7
            (4, 3), # 8
            (5, 3), # 9
            (1, 2), # 10
            (4, 2), # 11
            (0, 1), # 12
            (1, 1), # 13
            (2, 1), # 14
            (4, 1), # 15
            (5, 1), # 16
            (2, 0), # 17
            (5, 0), # 18
            ]
    ql = q.Q(states_to_states_with_rewards)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    trial_number = 0
    for i in range(1000):
        trial_number = i
        print('{0}: '.format(i), end='')
        ql.print_action_value_function(2)
        if i % 100 == 0:
            callback = view
        else:
            callback = None
        ql.epsilon_greedy(0.1, 0.1, 0.9, start_state=START_STATE, callback=callback)
    print('result: ', end='')
    ql.print_action_value_function(2)

if __name__ == '__main__':
    main()
