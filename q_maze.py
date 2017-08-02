# coding: utf-8

import sys
import pygame
from pygame.locals import Rect, QUIT
from moyf.ml.rl import q

WIDTH = 600
HEIGHT = 300

def main():
    def view(agent_state):
        pygame.display.set_caption("Q-Learning trial: {0}".format(trial_number))
        screen.fill((0, 0, 0))

        for i, position in enumerate(positions):
            left_top = (position[0] * WIDTH / 6, position[1] * HEIGHT / 3)
            pygame.draw.rect(screen, (255, 255, 255), Rect(left_top[0], left_top[1], WIDTH / 6, HEIGHT / 3))
            if i == agent_state:
                pygame.draw.rect(screen, (0, 0, 255), Rect(left_top[0], left_top[1], WIDTH / 6, HEIGHT / 3))

        pygame.display.update()
        pygame.time.wait(100)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        

    states_to_states_with_rewards = [
            [[1, 0]], # 0
            [[0, 0], [2, 0]], # 1
            [[1, 0], [3, 0], [5, 0]], # 2
            [[2, 0], [4, 0]], # 3
            [[3, 0], [6, 0]], # 4
            [[2, 0], [8, 0]], # 5
            [[4, 0], [9, 0]], # 6
            [[8, 0]], # 7
            [[5, 0], [7, 0]], # 8
            [[6, 0], [10, 100]], # 9
            [] # 10
            ]
    positions = [
            (0, 2), # 0
            (1, 2), # 1
            (2, 2), # 2
            (3, 2), # 3
            (4, 2), # 4
            (2, 1), # 5
            (4, 1), # 6
            (1, 0), # 7
            (2, 0), # 8
            (4, 0), # 9
            (5, 0), # 10
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
        ql.epsilon_greedy(0.05, 0.1, 0.9, callback=callback)
    print('result: ', end='')
    ql.print_action_value_function(2)

if __name__ == '__main__':
    main()
