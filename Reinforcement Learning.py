from TicTacToe import TicTacToe, GameStatus
from Connect4 import Connect4
import datetime
import json
import numpy as np
import random
from tensorflow import keras

# Training is split into 4 phases: monte carlo, temporal difference, value iteration, and deep q learning
mc_episodes, td_episodes, vi_episodes = 10_000, 10_000, 10_000
mc_epsilon, td_epsilon = [0, 50], [25, 75]
td_steps = 5
learning_rate = 0.01
discount_factor = 0.9
QTable = dict() # hashed state: [next states, next state values']

def best_action(next_states, player):
    action = None
    best_value = -100
    sign = 2 - 3 * player
    for i, state in enumerate(next_states):
        if state:
            value = (sign * QTable[state]).min()
            if value > best_value:
                action, best_value = i, value
    return action

def generate_episode(epsilon):
    state = TicTacToe()
    history = []
    player = 1

    game_status = GameStatus.ONGOING
    while game_status == GameStatus.ONGOING:
        next_states = QTable.get(hash(state), [None])[0]
        if random.random() < epsilon or not next_states:
            action = random.choice(state.actions)
        else:
            action = best_action(next_states, player)

        history.append((hash(state), action))
        game_status = state.move(action, player)
        player = 2 if player == 1 else 1
    
    return history, game_status

def td_lambda_qtable_update(history, game_status, n_step):
    next_state = history[-1]
    value = {GameStatus.X_WINS: 10, GameStatus.O_WINS: -10, GameStatus.TIE: 0}[game_status]
    sign = (len(history) % 2)
    for state, action in history[-2::-1]:

        q_row = QTable.setdefault(state, [[0] * 9, np.full([9], sign * 100, dtype = np.float32)])
        q_row[0][action] = next_state
        q_row[1][action] = value if q_row[1][action] == 100 else q_row[1][action] + learning_rate * (value - q_row[1][action])

        next_state = state
        value *= discount_factor
        sign = -sign

# PHASE 1: Monte Carlo Learning

def mc_qtable_update(history, game_status):
    td_lambda_qtable_update(history, game_status)

for episode_num in range(mc_episodes):
    history, game_status = generate_episode(round(episode_num / mc_episodes, 2))
    mc_qtable_update(history, game_status)

def td_qtable_update():
    pass

def vi_qtable_update():
    pass

# generates episodes of a given state and returns the history of the actions; uses an epsilon-greedy policy
def evaluate_the_policy():
    global state
    history = []
    descendants = state.descendants
    epsilon = episode_count / episodes
    isAgentXTurn = True

    while len(descendants):
        temp = [key for key, val in descendants.items() if val[0] == None]
        if random.random() < epsilon and len(temp) < len(descendants):
            action = (self_agent_action(state, steps) if isAgentXTurn else other_agent_action(state, steps))[0]
        else:
            actions = list(state.descendants.keys()) if epsilon > 0.5 or len(temp) == 0 else temp
            action = random.choice(actions)

        history.append((state, action))
        state = state.move(action, 2 - int(isAgentXTurn))
        descendants = state.descendants
        isAgentXTurn = not isAgentXTurn

    return history

# improves the QTable from the history of an episode and logs the result; uses a geometric backtrack
def improve_the_policy(history):
    global CurrScore, state
    if len(history) < 9 or state.reward: # either the game is won/lost and ends early or the game finishes with a win/loss and there is a nonzero reward
        CurrScore[1 - np.sign(state.reward)] += 1
    else:
        CurrScore[1] += 1

    reward = state.reward
    for trace, action in history[::-1]:
        value_past = trace.descendants[action][1]
        value_new = reward = gamma * reward + trace.reward
        trace.descendants[action][1] = value_past + alpha * (value_new - value_past) if trace.descendants[action][0] == None else value_new
    state = trace

# PHASE 2: Temporal Difference Learning

# PHASE 3: Value Iteration Learning

