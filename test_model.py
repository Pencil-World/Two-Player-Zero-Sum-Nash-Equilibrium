from model_manager import *
from TicTacToe import TicTacToe, GameStatus
import random

# def symmetry_expansion(state, old_action, player):
#     lowest_hash = state.symmetry_reduction() + player * (3 ** old_action)
#     actions = [[0, 2, 6, 8], [1, 3, 5, 7]][old_action % 2]
#     for new_action in actions:
#         board = TicTacToe(state)
#         board.move(new_action, player)
#         if lowest_hash == hash(board):
#             return new_action

def play_game(QTable):
    user = int(input("Enter which player you want to play as (1 for X, 2 for O): "))
    state = TicTacToe()
    player = 1

    game_status = GameStatus.ONGOING
    while game_status == GameStatus.ONGOING:
        if player == user:
            print('\n' + str(state))
            action = None
            while action not in state.all_valid_actions:
                action = int(input("Input move: "))
        else:
          next_states = QTable.get(state.symmetry_reduction(), [None])[0]
          if not next_states:
              action = random.choice(state.get_actions)
              print("Unknown board state")
          else:
              action = best_action(QTable, next_states, player)
              if action != 4:
                  action = symmetry_expansion(state, action, player)

        game_status = state.move(action, player)
        player = 2 if player == 1 else 1

    print('\n' + str(state))          
    match game_status:
        case GameStatus.X_WINS:
            print("X WINS")
        case GameStatus.O_WINS:
            print("O WINS")
        case GameStatus.TIE:
            print("TIE")

QTable = download_data()["QTable"]
while True:
    play_game(QTable)
