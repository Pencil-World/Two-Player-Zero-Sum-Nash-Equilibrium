from TicTacToe import TicTacToe
# from Connect4 import Connect4
import datetime
import json
from tensorflow import keras
import numpy as np
import random

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

"""
def Conclude():
    global BestScore
    pi = epoch_count % 10 if len(test_stats) > 1 else 0
    test_stats[pi][1] = 1.25 * CurrScore[0] + CurrScore[1]
    if BestScore[1] < test_stats[pi][1]:
        BestScore = [gamma, test_stats[pi][1]]
        JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()])) # works for jagged arrays. includes commas
        json.dump(JSON, open('agent.json', 'w'), indent = 4)

def log(text):
    with open('log.txt', 'a') as log:
        log.write(text + "\n")

def self_agent_action(begin, end):
    if begin == end:
        return
    return max(self_agent_action(begin, end)) if begin else min(self_agent_action(begin, end))

def other_agent_action(depth):
    return other_agent_action(depth - 1)

def evaluate_the_policy():
    actions = state.actions
    while actions.shape[0] and state.progress == "GREEN":
        blind = True
        values = QTable.setdefault(repr(state), np.full([9, 9], 100, dtype = np.float32)) if PlayerXTurn else values[action]
        epsilon = (epsilon / episode_count) * 100
        if random.randrange(0, 100) < epsilon:
            if PlayerXTurn:
                action =  np.array([-100 if min(elem) == 100 else min(elem) for elem in values]).argmax()
                blind = values[action].argmin() == 100
                # when less than halfway through. random actions will always choose an unexplored option
                # when over halfway through, weight the actions appropiately. maybe not weight the actions and do a standard random
                # the opponent will play in the same function
            else:
                action = values.argmin()
                blind = values[action] == 100

        if blind:
            if not PlayerXTurn and (9 - actions.shape[0]) < (values == 100).sum():
                action = actions[values[actions].argmax()]
            else:
                action = actions[random.randrange(0, actions.shape[0])]

        history.append((action, values[action]))
        state.move(action, PlayerXTurn + 1)
        actions = state.generate()
        PlayerXTurn = not PlayerXTurn

def improve_the_policy():
    pass

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

#open('The One.json', 'w').write(open('agent.json').read())
#Test()
open('log.txt', 'w').close()

# Variables are in alphabetical order
alpha = 0.001 # learning rate. model learns faster at higher alpha and learns slower at lower alpha
depth = 2 # prediction depth. int for how many moves in the future the model considers
episodes = 1000 # training time. model spends more time learning at higher episodes and spends less time training at lower episodes
epochs = 100 # evolutionary time. how many models and time spent exploring those models
gamma = 0.5 # decay rate. model considers future actions more at higher gamma and considers future actions less at lower gamma
state = TicTacToe()
test_stats = [[0.5, 1]]

BestScore = [0, 0]

log(f"starting program at {datetime.datetime.now()}")
for epoch_count in range(epochs):
    log(f"new model - starting epoch {epoch_count} at {datetime.datetime.now()}\n")
    #Experiment()

    CurrScore = np.zeros([3])
    QTable = dict()
    
    episode_count = 0
    while episode_count := episode_count + 1 < episodes:
        history = []
        PlayerXTurn = True;

        # mae this "learning" section a function as well
        it = int(PlayerXTurn) if state.progress == "RED" else 2
        if epsilon + 1 == epochs:
            CurrScore[it] += 1
            state = TicTacToe()
        else:
            reward = R[it]
            prev = 0
            for action, values in history[::-1]:
                PlayerXTurn = not PlayerXTurn
                state.move(action, 0)
                if PlayerXTurn:
                    reward *= gamma
                    values[prev] = reward if values[prev] == 100 else values[prev] + alpha * (reward - values[prev])
                else:
                    prev = action

        CurrScore /= episodes
        Conclude()
        log(f"{BestScore[0]:.3f} high score: {BestScore[1]:.3f}\t\t{gamma:.3f} win rate: {CurrScore[0]:.3f} loss rate: {CurrScore[1]:.3f} tie rate: {CurrScore[2]:.3f}\ttime: {datetime.datetime.now()}\n")