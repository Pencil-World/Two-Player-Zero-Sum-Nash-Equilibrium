from TicTacToe import TicTacToe
from Connect4 import Connect4
import datetime
from collections import deque
import json
import numpy as np
import random
from tensorflow import keras

# ***IMPORTANT*** NO EXPECTED RETURN NOR REWARD SHOULD BE GREATER THAN 100_000 EVER! 100_000 IS THE SAFE NUMBER!

'''
Please change the multiplier of Experiment.MoE to control the convergence rate of gamma. 
Tests each of the 10 test stats for gamma values
After testing each test stat, calculates the weighted average of the test stats
generates 10 test stats equally distributed across the margin of error
Each generation of test stats will decrease the margin of error until the convergence of gamma
'''
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

'''
Please change the score function depending on the scenario of the situation used. 

'''
def Conclude():
    global BestScore
    pi = epoch_count % 10 if len(test_stats) > 1 else 0
    test_stats[pi][1] = ScoreTicTacToe()
    if BestScore[1] < test_stats[pi][1]:
        BestScore = [gamma, test_stats[pi][1]]
        QTable = dict()
        queue = deque([state])

        while len(queue) > 0:
            item = queue.popleft()
            lst = [100_000] * 9
            for action, (trace, value) in item.children.items():
                if trace:
                    queue.append(trace)
                lst[action] = value
            QTable[repr(item)] = repr(lst)

        JSON = dict(zip(QTable.keys(), QTable.values())) # works for jagged arrays. includes commas
        json.dump(JSON, open('agent.json', 'w'), indent = 4)

def log(text):
    with open('log.txt', 'a') as log:
        log.write(text + "\n")

# returns the best action for the model given the state and foresight
def self_agent_action(state, _steps):
    if not _steps:
        return max(state.children.values(), key = lambda k: (k[1] if k[0] != None else -100_000))
    
    max_value_action = None
    for action, (child, junk) in state.children.items():
        if child:
            value = other_agent_action(child, _steps - 1)[1] if child.children else child.reward
            if not max_value_action or value > max_value_action[1]:
                max_value_action = (action, value)
    return max_value_action

# returns the best action for the opponent given the state and foresight
def other_agent_action(state, _steps):
    if not _steps:
        return min(state.children.values(), key = lambda k: (k[1] if k[0] != None else 100_000))
    
    min_value_action = None
    for action, (child, junk) in state.children.items():
        if child:
            value = self_agent_action(child, _steps - 1)[1] if child.children else child.reward
            if not min_value_action or value < min_value_action[1]:
                min_value_action = (action, value)
    return min_value_action

# generates episodes of a given state and returns the history of the actions; uses an epsilon-greedy policy
def evaluate_the_policy():
    global state
    history = []
    actions = state.actions
    isAgentXTurn = True

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
        state = state.move(action, 2 - int(isAgentXTurn))
        actions = state.actions
        isAgentXTurn = not isAgentXTurn
    
    return history

# improves the QTable from the history of an episode and logs the result; uses a geometric backtrack
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

# IMPROVE THE FETCHING SPEEDS OF THE ONE JSON BECAUSE IT'S SUPER SLOW RIGHT NOW
def Test():
    print("Test")
    QTable = json.load(open('The One.json', 'r'))
    
    root = TicTacToe()
    queue = deque([root])
    while len(queue) > 0:
        item = queue.popleft()
        values = json.loads(QTable.get(repr(item), "[100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]"))
        for action in item.actions:
            if values[action] < 100_000:
                trace = item.move(action)
                item.children[action] = [trace, values[action]]
                queue.append(trace)
        
    for temp in range(1_000):
        state = root
        isAgentX = AgentTurn = round(random.random())
        while state.actions:
            print('\n' + str(state))
            actions = state.actions
            if AgentTurn:
                if len([elem for elem in actions if state.children[elem][0] == None]) < len(state.children):
                    action = (self_agent_action(state, steps) if isAgentX else other_agent_action(state, steps))[0]
                else:
                    print("Unknown scenario")
                    action = actions[random.randrange(0, len(actions))]
            else:
                action = -1
                while not action in actions:
                    action = int(input("Input your turn: ")) - 1

            AgentTurn = not AgentTurn
            state = state.move(action)
        print('\n' + str(state))

def ScoreTicTacToe():
    return 1.25 * (CurrScore[0] - CurrScore[2]) + CurrScore[1]

def ScoreConnect4():
    return 2 * CurrScore[0] + CurrScore[1]

# Variables are in alphabetical order
alpha = 0.001 # learning rate. model learns faster at higher alpha and learns slower at lower alpha
steps = 1 # prediction steps. int for how many moves in the future the model considers
episodes = 5_000 # training time. model spends more time learning at higher episodes and spends less time training at lower episodes
epochs = 1_000 # evolutionary time. how many models and time spent exploring those models
gamma = 0.5 # decay rate. model considers future actions more at higher gamma and considers future actions less at lower gamma
state = TicTacToe()
test_stats = [[gamma, 1]]

# open('The One.json', 'w').write(open('agent.json').read())
# Test()
open('log.txt', 'w').close()

BestScore = [0, 0]
log(f"starting program at {datetime.datetime.now()}")
for epoch_count in range(epochs):
    Experiment()
    CurrScore = np.zeros([3])
    episode_count = 0
    while (episode_count := episode_count + 1) <= episodes:
        history = evaluate_the_policy()
        improve_the_policy(history)

    CurrScore /= episodes
    Conclude()
    log(f"{BestScore[0]:.3f} high score: {BestScore[1]:.3f}\t\tgamma: {gamma:.3f} win rate: {CurrScore[0]:.3f} tie rate: {CurrScore[1]:.3f} loss rate: {CurrScore[2]:.3f}\ttime: {datetime.datetime.now()}")