import numpy as np
from enum import Enum

class GameStatus(Enum):
    X_WINS = 1
    O_WINS = 2
    TIE = 0
    ONGOING = -1

class TicTacToe():
    table = [" ", "X", "O"]
    directions = np.array([[0, 1, 2], [0, 3, 6], [0, 2, 4], [0, 4, 8]]) # 4 different checks: horizontal, vertical, top right lower left, top left lower right
    
    def __init__(self, other = None, move = None):
        if isinstance(other, TicTacToe):
            self.board = other.board.copy()
            if move is not None:
                self.board[move[0]] = move[1]
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
    def all_actions(self):
        return [i for i, elem in enumerate(self.board) if elem == 0]

    # super convoluted; do not change
    @property
    def get_actions(self):
        if self.actions is None:
            state_map = set()
            self.actions = []
            for action in self.all_actions:
                board = TicTacToe(self, (action, 1))
                old_hash, new_hash = hash(board), board.symmetry_reduction()
                if new_hash not in state_map:
                    state_map.add(old_hash)
                    self.actions.append(action)
        return self.actions

    def move(self, action, player):
        self.board[action] = player
        self.actions = None
        self.hash = None
        return self.__evaluate(action, player)

    def __evaluate(self, action, player):
        bases = [(action // 3) * 3, action % 3, 2 if action in [2, 4, 6] else None, 0 if action in [0, 4, 8] else None]
        for base, direction in zip(bases, TicTacToe.directions):
            if base is not None:
                temp = direction + base
                if self.board[temp[0]] == self.board[temp[1]] == self.board[temp[2]]:
                    return GameStatus.X_WINS if player == 1 else GameStatus.O_WINS
        if self.get_actions == []:
            return GameStatus.TIE
        return GameStatus.ONGOING

    def __hash__(self):
        if self.hash is None:
            self.hash = sum(int(cell) * (3 ** i) for i, cell in enumerate(self.board))
        return self.hash

    def symmetry_reduction(self):
        board_copy = self.board.copy()
        lowest_hash = hash(self)
        #                 horizontal flip,       vertical flip,         rotate ccw,            rotate cw,             rotate 180,              diagonal flip,       anti-diagonal flip
        for transform in [lambda r, c: (r, 2-c), lambda r, c: (2-r, c), lambda r, c: (c, 2-r), lambda r, c: (2-c, r), lambda r, c: (2-r, 2-c), lambda r, c: (c, r), lambda r, c: (2-c, 2-r)]:
            i = 0
            for r in range(3):
                for c in range(3):
                    new_r, new_c = transform(r, c)
                    new_i = new_r * 3 + new_c
                    self.board[new_i] = board_copy[i]
                    i += 1

            if hash(self) < lowest_hash:
                lowest_hash = hash(self)
            self.hash = None

        self.board = board_copy
        return lowest_hash
