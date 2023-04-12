from Board import Board
from Connect4 import Connect4
import datetime
import json
import numpy as np
import random

def Experiment():
    global gamma
    pi = i % 10
    if pi == 0:
        gamma = HighScore[-1] # dot product average
        
        lower, upper = max(0, gamma - Experiment.MoE), min(gamma + Experiment.MoE, 1)
        delta = (upper - lower) / 9
        test_stats = [delta * elem + lower for elem in range(10)]
        Experiment.MoE *= 0.999
    gamma = test_stats[pi]
Experiment.MoE = 0.5

def Conclude():
    global HighScore
    if HighScore[0] < CurrScore[0]:
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
#def Test():
#    print("\nTest")
#    QTable = json.load(open('The Apex.json', 'r'))
#    for key, val in QTable.items():
#        QTable[key] = np.array(json.loads(val))

#    state = Board()
#    ModelTurn = True
#    for temp in range(1_000):
#        print('\n' + str(state))
#        action = int(input()) if ModelTurn else (QTable[repr(state)].argmax() if random.randrange(0, 100) < 95 and repr(state) in QTable else actions[random.randrange(0, actions.shape[0])])
#        state.move(action, ModelTurn + 1)

#        actions = state.generate()
#        ModelTurn = not ModelTurn
#        if not actions.shape[0] or state.progress == "RED":
#            print('\n' + str(state))
#            state = Board()
#            ModelTurn = True

alpha = 0.0001
gamma = 0.5
episodes = 1_000
data_size = 1_000

HighScore = [0, 0, 0]
R = [10, -10, 0]
bounds = [0, 1]

#open('The Apex.json', 'w').write(open('agent.json').read())
#Test()
open('log.txt', 'w').close()

state = Board()
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
                node = None
                if random.randrange(0, 100) < epoch:
                    action = None
                elif (len(history) % 2):                    
                    matrix = np.fromfunction(np.vectorize(lambda i, j: min(node[i])), [len(node), 1], dtype = int)

                action = actions[random.randrange(0, actions.shape[0])]
                action = node[min().argmax()].argmin()
                node = QTable.setdefault(repr(state), np.full([9, 9], -1000, dtype = np.float32))
                index = node.argmax() if random.randrange(0, 100) < epoch * lim else None
                action = index if index and node[index] != -1000 else action
                
                if actions.shape[0] and state.progress != "RED":
                    break
                history.append((action, node[action[0]]))
                state.move(action, isModel + 1)
                actions = state.generate()

            it = len(history) % 2 if state.progress == "RED" else 2
            reward = R[it]
            CurrScore[it] += epoch + 1 == lim

            prev = None
            for action, node in history[::-1]:
                isModel = not isModel
                state.move(action, 0)
                if isModel:
                    reward = R[3] + gamma * reward
                    if prev:
                        node[action][prev] = reward if node[action] == -1000 else node[action] + alpha * (reward - node[action])
                    else:
                        node[action] = np.full([1, 5], reward)
                else:
                    prev = action

    CurrScore /= episodes
    #open('log.txt', 'a').write(f"win rate: {CurrScore[0]:.3f}\tloss rate: {CurrScore[1]:.3f}\ttie rate: {CurrScore[2]:.3f}\n")
    Conclude()