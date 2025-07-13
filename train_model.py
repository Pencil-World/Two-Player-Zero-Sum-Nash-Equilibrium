from TicTacToe import TicTacToe, GameStatus
from Connect4 import Connect4
import numpy as np
import random
from model_manager import *
# from tensorflow import keras

# Training is split into 4 phases: monte carlo, temporal difference, value iteration, and deep q learning
discovery_episodes, mc_episodes, td_episodes, vi_episodes = 500, 500, 500, 5
mc_epsilon, td_epsilon = [1, 0.5], [0.5, 0]
# mc_epsilon, td_epsilon = [1, 0], [0.75, 0.25]
td_steps = 4
learning_rate = 0.01
discount_factor = 0.9
QTable = dict() # hashed state: [next states, next state values'] or [None, value]

def generate_episode(epsilon, discover = False):
    state = TicTacToe()
    history = []
    player = 1
    # print("new game")

    game_status = GameStatus.ONGOING
    while game_status == GameStatus.ONGOING:
        next_states = QTable.get(hash(state), [None])[0]
        if random.random() < epsilon or not next_states:
            actions = state.get_actions
            if discover and next_states:
                unexplored_states = [i for i, val in enumerate(next_states) if val == 0]
                actions = list(set(actions) & set(unexplored_states)) or actions
            action = random.choice(actions)
        else:
            action = best_action(QTable, next_states, player)

        history.append((hash(state), action))
        game_status = state.move(action, player)
        player = 2 if player == 1 else 1
        # print(str(state) + '\n')
    
    history.append(hash(state))
    return history, game_status

# n_step is None means using monte carlo
def td_lambda_qtable_update(history, game_status, n_step):
    next_state = history[-1]
    values = [{GameStatus.X_WINS: 10, GameStatus.O_WINS: -10, GameStatus.TIE: 0}[game_status]]
    sign = 2 * (len(history) % 2) - 1

    QTable[next_state] = [None, np.array(values[-1])]
    for state, action in history[-2::-1]:
        q_row = QTable.setdefault(state, [[0] * 9, np.full([9], sign * 100, dtype = np.float32)])
        temp = values[-1] if abs(q_row[1][action]) == 100 else q_row[1][action]
        q_row[0][action] = next_state
        q_row[1][action] = temp + learning_rate * (values[-1] - temp)

        next_state = state
        if n_step:
            values.append((values[-1] if len(values) < n_step else values[-n_step]) * discount_factor)
            values[-2] = temp
        else:
            values[-1] *= discount_factor
        sign = -sign

# PHASE 1: Discovery

def mc_qtable_update(history, game_status):
    td_lambda_qtable_update(history, game_status, None)

def td_qtable_update(history, game_status):
    td_lambda_qtable_update(history, game_status, td_steps)

for episode_num in range(discovery_episodes):
    history, game_status = generate_episode(1, True)
    td_qtable_update(history, game_status)

# PHASE 1: Monte Carlo Learning

for episode_num in range(mc_episodes):
    history, game_status = generate_episode(round(np.interp(episode_num, [0, mc_episodes - 1], mc_epsilon), 2))
    mc_qtable_update(history, game_status)

# PHASE 2: Temporal Difference Learning

for episode_num in range(td_episodes):
    history, game_status = generate_episode(round(np.interp(episode_num, [0, td_episodes - 1], td_epsilon), 2))
    td_qtable_update(history, game_status)

# PHASE 3: Value Iteration Learning

def vi_qtable_update():
    pass

upload_data(QTable=QTable, discovery_episodes=discovery_episodes, mc_episodes=mc_episodes, td_episodes=td_episodes, mc_epsilon=mc_epsilon, td_epsilon=td_epsilon, td_steps=td_steps, learning_rate=learning_rate, discount_factor=discount_factor)
