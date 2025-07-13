import json
import numpy as np

def best_action(QTable, next_states, player):
    action = None
    best_value = -100
    sign = 3 - 2 * player
    for i, state in enumerate(next_states):
        if state:
            value = (sign * QTable[state][1]).min()
            if value > best_value:
                action, best_value = i, value
    return action

discovery_episodes, mc_episodes, td_episodes, vi_episodes = 500, 500, 500, 5
mc_epsilon, td_epsilon = [1, 0.5], [0.5, 0]
# mc_epsilon, td_epsilon = [1, 0], [0.75, 0.25]
td_steps = 4
learning_rate = 0.01
discount_factor = 0.9
QTable = dict() # hashed state: [next states, next state values'] or [None, value]

def upload_data(discovery_episodes=None, mc_episodes=None, td_episodes=None, mc_epsilon=None, td_epsilon=None, td_steps=None, learning_rate=None, discount_factor=None, QTable=None):
    for key, val in QTable.items():
        QTable[key][1] = val[1].tolist()
    agent_data = {"discovery_episodes": discovery_episodes, "mc_episodes": mc_episodes, "td_episodes": td_episodes, "mc_epsilon": mc_epsilon, "td_epsilon": td_epsilon, "td_steps": td_steps, "learning_rate": learning_rate, "discount_factor": discount_factor, "QTable": QTable}
    with open("agent.json", "w") as f:
        json.dump(agent_data, f, indent=2)

def download_data():
    with open("agent.json", "r") as f:
        agent_data = json.load(f)
        QTable = agent_data["QTable"]
        for key, val in QTable.items():
            QTable[key][1] = np.array(val[1])
        return agent_data