# coding: utf-8

import sys
import pygame
from pygame.locals import Rect, QUIT
from moyf.ml.rl import reinforce

# ground, hole, ground, hole, ...
ENVIRONMENT = '####### ##### ##### ######  ### #### #####  #######'
SPEED = 0.5
JUMP_SPEED = 30.0
GRAVITY = 9.8

WIDTH = 600
HEIGHT = 300

trial_number = 0
screen = None

def expected_values_callback(state, theta):
    if state['is_jumping']:
        return [1.0, 0.0]
    return [theta_row[0] * (state['hole_start'] - state['x']) + theta_row[1] for theta_row in theta]
    #return (state['hole_start'] - state['x']) * theta

def calc_state_reward_callback(action, state):
    if trial_number % 100 == 0:
        pygame.display.set_caption("REINFORCE-Algorithm trial: {0}".format(trial_number))
        screen.fill((255, 255, 255))

        rect_size = WIDTH / len(ENVIRONMENT)
        for i, c in enumerate(ENVIRONMENT):
            if c == '#':
                pygame.draw.rect(screen, (0, 0, 0), Rect(i * rect_size, HEIGHT * 0.75, rect_size + 1, rect_size))

        agent_size = rect_size * 0.25
        pygame.draw.ellipse(screen, (0, 0, 255), Rect(state['x'] * rect_size, HEIGHT * 0.75 - agent_size - state['y'] * rect_size, agent_size, agent_size))

        font = pygame.font.Font(None, 30)
        screen.blit(font.render('action: {0}'.format(action), True, (0, 0, 0)), [0, 0])

        pygame.display.update()
        pygame.time.wait(20)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

    state['x'] += SPEED
    state['time'] += 1.0
    if state['is_jumping']:
        time_delta = state['time'] - state['jump_start_time']
        tmp_y = JUMP_SPEED * time_delta - GRAVITY * (time_delta ** 2) * 0.5
        if tmp_y <= 0:
            state['y'] = 0
            state['is_jumping'] = False
#            print('foo tmp_y = {0}'.format(tmp_y))
        else:
            state['y'] = tmp_y
#            print('bar tmp_y = {0}'.format(tmp_y))
    elif action == 1:
        state['is_jumping'] = True
        state['jump_start_time'] = state['time']

    if state['x'] >= state['hole_start'] and state['y'] <= 0:
        return state, -100, True
    if state['x'] >= state['ground_start']:
        state['hole_start'] = ENVIRONMENT.find(' ', int(state['x']))
        state['ground_start'] = ENVIRONMENT.find('#', state['hole_start'])

    return state, state['x'], False

def main():
    rf = reinforce.REINFORCE(
            expected_values_callback,
            reinforce.REINFORCE.decide_action,
            calc_state_reward_callback,
            [2, 2],
            reinforce.REINFORCE.policy_softmax
            )
    start_state={
            'x': 0.0,
            'y': 0,
            'time': 0,
            'is_jumping': False,
            'jump_start_time': 0,
            'hole_start': ENVIRONMENT.find(' '),
            }
    start_state['ground_start'] = ENVIRONMENT.find('#', start_state['hole_start'])

    pygame.init()
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    for i in range(10000):
        global trial_number
        trial_number = i
        if trial_number % 100 == 0:
            print('{0}: {1}'.format(i, rf.theta))
        rf.update(start_state=start_state, learning_rate=0.1, episode_num=1)
    print('result: {0}'.format(rf.theta))

if __name__ == '__main__':
    main()
