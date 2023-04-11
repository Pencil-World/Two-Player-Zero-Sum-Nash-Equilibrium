from Board import Board
from Connect4 import Connect4
import datetime
import json
import numpy as np
import random

def Clear():
    open('log.txt', 'w').close()

def Experiment():
    global gamma
    pi = i % 10
    if pi == 0:
        gamma = HighScore[-1]
        lower, upper = max(0, gamma - 0.5 * 0.999**i), min(gamma + 0.5 * 0.999**i, 1)
        delta = (upper - lower) / 9
        test_stats = [delta * elem + lower for elem in range(10)]
    gamma = test_stats[pi]

def NewRecord():
    global HighScore
    with open('debugger.txt', 'a') as debugger:
        debugger.write(f"{HighScore[0]:.3f}-{CurrScore[0]:.3f}\tgamma: {gamma:.3f}\twin reward: {R[0]}\tloss reward: {R[1]}\ttie reward: {R[2]}\tdefault reward: {R[3]}\n")
    HighScore = CurrScore
    JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()])) # works for jagged arrays. includes commas
    json.dump(JSON, open('agent.json', 'w'), indent = 4)

def Print():
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")
        debugger.write(f"time: {datetime.datetime.now()}\ti: {i}\tbounds: {str(bounds.tolist())}")

# fix pls
def Test():
    print("\nTest")
    QTable = json.load(open('The Apex.json', 'r'))
    for key, val in QTable.items():
        QTable[key] = np.array(json.loads(val))

    state = Board()
    ModelTurn = True
    for temp in range(1_000):
        print('\n' + str(state))
        action = int(input()) if ModelTurn else (QTable[repr(state)].argmax() if random.randrange(0, 100) < 95 and repr(state) in QTable else actions[random.randrange(0, actions.shape[0])])
        state.move(action, ModelTurn + 1)

        actions = state.generate()
        ModelTurn = not ModelTurn
        if not actions.shape[0] or state.progress == "RED":
            print('\n' + str(state))
            state = Board()
            ModelTurn = True

alpha = 0.0001
gamma = 0.5
episodes = 1_000
data_size = 1_000

HighScore = np.zeros([3])
R = 10
bounds = [0, 1]

#open('The Apex.json', 'w').write(open('agent.json').read())
#Test()
open('log.txt', 'w').close()

state = Connect4()
lim = 10
with open('log.txt', 'a') as log:
    log.write(f"start program\n{datetime.datetime.now()}\n")
for i in range(data_size):
    target = Experiment()
    CurrScore = np.zeros([3])
    QTable = dict()
    for epoch in range(lim):
        for temp in range(episodes):
            history = []
            actions = state.generate()

            while actions.shape[0] and state.progress != "RED":
                strategies = min().argmax()
                mean = None
                # matrix = np.fromfunction(np.vectorize(lambda i, j: min(mean[i])), [len(mean), 1], dtype = int)

                action = actions[random.randrange(0, actions.shape[0])]
                action = mean[min().argmax()].argmin()
                mean = QTable.setdefault(repr(state), np.full([9, 9], -1000, dtype = np.float32))
                index = mean.argmax() if random.randrange(0, 100) < epoch * lim else None
                action = index if index and mean[index] != -1000 else action

                history.append((action, mean[action[0]]))
                state.move(action, isModel + 1)
                actions = state.generate()

            it = len(history) % 2 if state.progress == "RED" else 2
            reward = R if it == 0 else -R
            CurrScore[it] += epoch + 1 == lim

            prev = None
            for action, mean in history[::-1]:
                isModel = not isModel
                state.move(action, 0)
                if isModel:
                    reward = R[3] + gamma * reward
                    if prev:
                        mean[action][prev] = reward if mean[action] == -1000 else mean[action] + alpha * (reward - mean[action])
                    else:
                        mean[action] = np.full([1, 5], reward)
                else:
                    prev = action

    CurrScore /= episodes
    open('log.txt', 'a').write(f"win rate: {CurrScore[0]:.3f}\tloss rate: {CurrScore[1]:.3f}\ttie rate: {CurrScore[2]:.3f}\n")
    if HighScore[0] < CurrScore[0]:
        NewRecord()