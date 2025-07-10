import numpy as np
from enum import Enum

class GameStatus(Enum):
    X_WINS = 1
    O_WINS = 2
    TIE = 0
    ONGOING = -1

class TicTacToe():
    table = [" ", "X", "O"]
    cipher = [1, 3, 9]
    decode = {1: 0, 3: 1, 9: 2}
    
    def __init__(self, other = None):
        if isinstance(other, TicTacToe):
            self.board = other.board.copy()
        elif isinstance(other, int):
            self.board = np.empty([9], dtype = np.int8)
            for i in range(9):
                self.board[i] = other % 3
                other //= 3
        else:
            self.board = np.full([9], 0, dtype = np.int8)
        self.actions = None
        self.hash = None

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = TicTacToe.table[elem]
        return str(board)

    @property
    def get_actions(self):
        if self.actions is None:
            self.actions = [i for i, elem in enumerate(self.board) if elem == 0]
        return self.actions

    def move(self, action, player):
        self.board[action] = player
        return self.__evaluate(action, player)

    def __evaluate(self, action, player):
        for direction in [[1, 2], [3, 6], [4, 8], [2, 4]]:
            total = 0
            for offset in direction:
                total += sum([TicTacToe.cipher[self.board[i]] for i in [action - offset, action + offset] if 0 <= i < 9])
            if total == 3 * TicTacToe.cipher[player]:
                return GameStatus.X_WINS if player == 1 else GameStatus.O_WINS
            
        if self.get_actions == []:
            return GameStatus.TIE
        return GameStatus.ONGOING

    def __hash__(self):
        if self.hash is None:
            self.hash = sum(TicTacToe.cipher[elem] for elem in self.board)
        return self.hash
