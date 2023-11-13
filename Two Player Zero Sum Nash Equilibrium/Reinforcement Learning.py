from TicTacToe import TicTacToe
# from Connect4 import Connect4
import datetime
from collections import deque
import json
import numpy as np
import random
from tensorflow import keras

"""

"""
def Experiment():
    global gamma, test_stats
    pi = epoch_count % 10
    if pi == 0:
        gamma = sum([elem[0] * elem[1] ** 5 for elem in test_stats]) / sum([elem[1] ** 5 for elem in test_stats])
        lower, upper = max(0.001, gamma - Experiment.MoE), min(gamma + Experiment.MoE, 0.999)
        delta = (upper - lower) / 9
        test_stats = [[delta * elem + lower, 0] for elem in range(10)]
        Experiment.MoE *= 0.9
    gamma = test_stats[pi][0]
Experiment.MoE = 0.5

"""
Please change the score function depending on the scenario of the situation used. 
"""
def Conclude():
    global BestScore
    pi = epoch_count % 10 if len(test_stats) > 1 else 0
    test_stats[pi][1] = ScoreTicTacToe()
    if BestScore[1] < test_stats[pi][1]:
        BestScore = [gamma, test_stats[pi][1]]
        QTable = dict
        queue = deque(state)

        while len(queue) > 0:
            item = queue.popleft()
            for state, value in item.children:
                QTable[repr(state)] = value
                queue.append(state)

        JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()])) # works for jagged arrays. includes commas
        json.dump(JSON, open('agent.json', 'w'), indent = 4)

def log(text):
    with open('log.txt', 'a') as log:
        log.write(text + "\n")


# use actions, not children, because there are invalid thingymabobs
def self_agent_action(state, _steps):
    if not _steps:
        return max(state.children.values(), key = lambda k: (k[1] if k[0] else -100_000))
    
    max_value_action = None
    for action, (child, junk) in state.children.items():
        if child:
            value = other_agent_action(child, _steps - 1)[1]
            if not max_value_action or value > max_value_action[1]:
                max_value_action = (action, value)
    return max_value_action

def other_agent_action(state, _steps):
    if not _steps:
        return min(state.children.values(), key = lambda k: (k[1] if k[0] else 100_000))
    
    min_value_action = None
    for action, (child, junk) in state.children.items():
        if child:
            value = self_agent_action(child, _steps - 1)[1]
            if not min_value_action or value < min_value_action[1]:
                min_value_action = (action, value)
    return min_value_action

def evaluate_the_policy():
    global state
    history = []
    actions = state.actions
    
    while len(actions):
        epsilon = episode_count / episodes
        temp = [elem for elem in actions if state.children[elem][0] == None]
        if random.random() < epsilon and len(temp) < len(state.children):
            action = (self_agent_action(state, steps) if len(history) % 2 else other_agent_action(state, steps))[0]
        else:
            if epsilon < 0.5:
                actions = state.actions if len(temp) == 0 else temp
            action = random.choice(actions)

        history.append((state, action))
        blah = state.move(action)
        if not isinstance(blah, TicTacToe):
            print()
        state = blah
        actions = state.actions
    
    return history

def improve_the_policy(history):
    global CurrScore, state
    if len(history) < 9 or state.reward:
        CurrScore[1 - np.sign(state.reward)] += 1
    else:
        CurrScore[1] += 1

    reward = state.reward
    for trace, action in history[::-1]:
        value_past = trace.children[action][1]
        value_new = reward = gamma * reward + trace.reward
        trace.children[action][1] = value_past + alpha * (value_new - value_past) if trace.children[action][1] else value_new
    state = trace

#def Test():
#    print("\nTest")
#    QTable = json.load(open('The One.json', 'r'))
#    for key, val in QTable.items():
#        QTable[key] = np.array(json.loads(val))

#    state = TicTacToe()
#    PlayerXTurn = True;
    
#    for temp in range(1_000):
#        if PlayerXTurn:
#            blind = True
#            if repr(state) in QTable:
#                values = QTable[repr(state)]
#                action = np.array([-100 if min(elem) == 100 else min(elem) for elem in values]).argmax()
#                blind = values[action].argmin() == 100
#            if blind:
#                print("unknown scenario")
#                action = actions[random.randrange(0, actions.shape[0])]
#        else:
#            action = int(input())

#        state.move(action, PlayerXTurn + 1)
#        print('\n' + str(state))

#        actions = state.generate()
#        PlayerXTurn = not PlayerXTurn
#        if not actions.shape[0] or state.progress == "RED":
#            state = TicTacToe()
#            PlayerXTurn = True

def ScoreTicTacToe():
    return 1.25 * (CurrScore[0] - CurrScore[2]) + CurrScore[1]

def ScoreConnect4():
    return 2 * CurrScore[0] + CurrScore[1]

#open('The One.json', 'w').write(open('agent.json').read())
#Test()
open('log.txt', 'w').close()

# Variables are in alphabetical order
alpha = 0.001 # learning rate. model learns faster at higher alpha and learns slower at lower alpha
steps = 1 # prediction steps. int for how many moves in the future the model considers
episodes = 1000 # training time. model spends more time learning at higher episodes and spends less time training at lower episodes
epochs = 100 # evolutionary time. how many models and time spent exploring those models
gamma = 0.5 # decay rate. model considers future actions more at higher gamma and considers future actions less at lower gamma
state = TicTacToe()
test_stats = [[gamma, 1]]

BestScore = [0, 0]
log(f"starting program at {datetime.datetime.now()}")
for epoch_count in range(epochs):
    log(f"new model - starting epoch {epoch_count} at {datetime.datetime.now()}\n")
    #Experiment()

    CurrScore = np.zeros([3])
    episode_count = 0
    while (episode_count := episode_count + 1) <= episodes:
        history = evaluate_the_policy()
        improve_the_policy(history)

    CurrScore /= episodes
    Conclude()
    log(f"{BestScore[0]:.3f} high score: {BestScore[1]:.3f}\t\t{gamma:.3f} win rate: {CurrScore[0]:.3f} tie rate: {CurrScore[1]:.3f} loss rate: {CurrScore[2]:.3f}\ttime: {datetime.datetime.now()}\n")