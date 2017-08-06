# coding: utf-8

from moyf.ml.rl import reinforce

# ground, hole, ground, hole, ...
ENVIRONMENT = '####### ##### ##### ######  ### #### #####  #######'
SPEED = 0.5
JUMP_SPEED = 15.0
GRAVITY = 9.8

def expected_values_callback(state, theta):
    if state['is_jumping']:
        return [1.0, 0.0]
    return (state['hole_start'] - state['x']) * theta

def calc_state_reward_callback(action, state, theta):
    state['x'] += SPEED
    state['time'] += 1.0
    if state['is_jumping']:
        time_delta = state['time'] - state['jump_start_time']
        tmp_y = SPEED * time_delta - GRAVITY * (time_delta ** 2) * 0.5
        if tmp_y <= 0:
            state['y'] = 0
            state['is_jumping'] = False
        else:
            state['y'] = tmp_y
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
            [2, 1],
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
    for i in range(100):
        print('{0}: {1}'.format(i, rf.theta))
        rf.update(start_state=start_state, learning_rate=0.1, episode_num=1)
    print('result: {0}'.format(rf.theta))

if __name__ == '__main__':
    main()
