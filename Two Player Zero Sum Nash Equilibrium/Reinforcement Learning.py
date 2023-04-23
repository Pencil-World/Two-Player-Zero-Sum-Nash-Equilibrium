from TicTacToe import TicTacToe
# from Connect4 import Connect4
import datetime
import json
from tensorflow import keras
import numpy as np
import random

def Experiment():
    global gamma, test_stats
    pi = i % 10
    if pi == 0:
        gamma = sum([elem[0] * elem[1] ** 5 for elem in test_stats]) / sum([elem[1] ** 5 for elem in test_stats])
        lower, upper = max(0.001, gamma - Experiment.MoE), min(gamma + Experiment.MoE, 0.999)
        delta = (upper - lower) / 9
        test_stats = [[delta * elem + lower, 0] for elem in range(10)]
        Experiment.MoE *= 0.9
    gamma = test_stats[pi][0]
Experiment.MoE = 0.5

def Conclude():
    global HighScore
    pi = i % 10
    test_stats[pi][1] = CurrScore[0]
    if HighScore[1] < CurrScore[0]:
        HighScore = [gamma, CurrScore[0]]
        JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()])) # works for jagged arrays. includes commas
        json.dump(JSON, open('agent.json', 'w'), indent = 4)

def Test():
    print("\nTest")
    QTable = json.load(open('The One.json', 'r'))
    for key, val in QTable.items():
        QTable[key] = np.array(json.loads(val))

    state = TicTacToe()
    AgentTurn = True;
    for temp in range(1_000):
        if AgentTurn:
            blind = True
            if repr(state) in QTable:
                values = QTable[repr(state)]
                action = np.array([-100 if min(elem) == 100 else min(elem) for elem in values]).argmax()
                blind = values[action].argmin() == 100
            if blind:
                print("unknown scenario")
                action = actions[random.randrange(0, actions.shape[0])]
        else:
            action = int(input())

        state.move(action, AgentTurn + 1)
        print('\n' + str(state))

        actions = state.generate()
        AgentTurn = not AgentTurn
        if not actions.shape[0] or state.progress == "RED":
            state = TicTacToe()
            AgentTurn = True

alpha = 0.0001
gamma = 0.5
episodes = 1000
epochs = 100

R = [10, -10, 0]
test_stats = [[0.5, 1]]
HighScore = [0, 0]
data_size = 1_000

#open('The One.json', 'w').write(open('agent.json').read())
#Test()
open('log.txt', 'w').close()

state = TicTacToe()
with open('log.txt', 'a') as log:
    log.write(f"start program\n{datetime.datetime.now()}\n")
for i in range(data_size):
    Experiment()
    CurrScore = np.zeros([3])
    QTable = dict()
    for epsilon in range(epochs):
        for temp in range(episodes):
            history = []
            actions = state.generate()
            AgentTurn = True;

            while actions.shape[0] and state.progress == "GREEN":
                blind = True
                values = QTable.setdefault(repr(state), np.full([9, 9], 100, dtype = np.float32)) if AgentTurn else values[action]
                if random.randrange(0, 100) < epsilon * (100 // epochs):
                    if AgentTurn:
                        action =  np.array([-100 if min(elem) == 100 else min(elem) for elem in values]).argmax()
                        blind = values[action].argmin() == 100
                    elif epsilon + 1 == epochs:
                        blind = False
                        size = actions.shape[0] // 4
                        action = actions[values.argpartition(size)[random.randint(0, size)]]
                    else:
                        action = values.argmin()
                        blind = values[action] == 100

                if blind:
                    action = actions[values[actions].argmax() if not AgentTurn and (9 - actions.shape[0]) < (values == 100).sum() else random.randrange(0, actions.shape[0])]

                history.append((action, values[action]))
                state.move(action, AgentTurn + 1)
                actions = state.generate()
                AgentTurn = not AgentTurn

            it = int(AgentTurn) if state.progress == "RED" else 2
            if epsilon + 1 == epochs:
                CurrScore[it] += 1
                state = TicTacToe()
            else:
                reward = R[it]
                prev = 0
                for action, values in history[::-1]:
                    AgentTurn = not AgentTurn
                    state.move(action, 0)
                    if AgentTurn:
                        reward *= gamma
                        values[prev] = reward if values[prev] == 100 else values[prev] + alpha * (reward - values[prev])
                    else:
                        prev = action

    CurrScore /= episodes
    Conclude()
    with open('log.txt', 'a') as log:
        log.write(f"{HighScore[0]:.3f} high score: {HighScore[1]:.3f}\t\t{gamma:.3f} win rate: {CurrScore[0]:.3f} loss rate: {CurrScore[1]:.3f} tie rate: {CurrScore[2]:.3f}\ttime: {datetime.datetime.now()}\n")