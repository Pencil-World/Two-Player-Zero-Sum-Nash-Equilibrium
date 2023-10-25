import numpy as np

# reward works like this: 1 for each x, 10 for each xx, 100 for each xxx. each must have the possibility for an xxx
class TicTacToe():
    def __init__(self, board = None):
        self.board = np.full([9], 0, dtype = np.int8)
        if board:
            for i, elem in enumerate(board):
                self.board[i] = elem

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        dict = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = dict[elem]
        return str(board)

    def move(self, action, player):
        self.board[action] = player

    @property
    def reward(self):
        tokenizer = [1, 3, 9] # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 7: 5, 9: 10, 19: -5, 27: -10 } # converts the tokens into rewards throughout the board
        reward = 0
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            temp = sum(tokenizer[self.board[[coord, coord + change, coord + change + change]]])
            reward += parser[temp]
        return reward

    @property
    def actions(self):
        actions = []
        for i, elem in enumerate(self.board):
            if not elem:
                actions.append(i)
        return np.array(actions, dtype = np.int8)