import numpy as np

class Connect4():
    def __init__(self, board = None):
        self.board = np.full([9], 0, dtype = np.int8)
        if board:
            for i, elem in enumerate(board):
                self.board[i] = elem

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([6, 7], "")
        dictionary = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 6][i % 7] = dictionary[elem]
        return str(board)

    def move(self, action, player):
        # incorrect move
        self.board[action] = player
        self.evaluate(action)

    def evaluate(self, action):
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            if 0 != self.board[coord] == self.board[coord + change] == self.board[coord + change + change]:
                self.progress = "RED"
                return
        self.progress = "GREEN"

    def generate(self):
        return np.array([elem for elem in [6 * elem + 5 for elem in range(7)] if self.board[elem] == 0], dtype = np.int8)