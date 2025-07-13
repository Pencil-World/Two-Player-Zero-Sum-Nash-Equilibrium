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

def upload_data(discovery_episodes=None, mc_episodes=None, td_episodes=None, mc_epsilon=None, td_epsilon=None, td_steps=None, learning_rate=None, discount_factor=None, QTable=None):
    print("upload data process starting")
    for key, val in QTable.items():
        QTable[key][1] = val[1].tolist()
    agent_data = {"discovery_episodes": discovery_episodes, "mc_episodes": mc_episodes, "td_episodes": td_episodes, "mc_epsilon": mc_epsilon, "td_epsilon": td_epsilon, "td_steps": td_steps, "learning_rate": learning_rate, "discount_factor": discount_factor, "QTable": QTable}
    with open("agent.json", "w") as f:
        json.dump(agent_data, f, indent=2)
    print("upload data process ending")

def download_data():
    print("download data process starting")
    with open("agent.json", "r") as f:
        agent_data = json.load(f)
        QTable = {}
        for key, val in agent_data["QTable"].items():
            QTable[int(key)] = [val[0], np.array(val[1])]
        agent_data["QTable"] = QTable
        return agent_data
    print("download data process ending")

def log(text):
    with open("log.txt", "append or whatever") as f:
        agent_data = json.load(f)
