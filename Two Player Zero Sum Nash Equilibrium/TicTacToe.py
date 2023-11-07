from asyncio.windows_events import NULL
import numpy as np

class TicTacToe():
    def __init__(self, board = None):
        self.board = np.full([9], 0, dtype = np.int8)
        if board:
            for i, elem in enumerate(board):
                self.board[i] = elem

        self.children = np.empty([9, 2], dtype = object)#contains children and their returns
        for i, elem in enumerate(self.board):
            self.children[i] = None if not elem else "INVALID"

        self.reward = self.__reward
        self.actions = self.actions

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
    def __reward(self):
        tokenizer = np.array([1, 3, 9]) # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 7: 5, 9: 10, 19: -5, 27: -10 } # converts the tokens into rewards throughout the board
        reward = 0
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            temp = sum(tokenizer[self.board[[coord, coord + change, coord + change + change]]])
            reward += parser[temp]
        return reward

    @property
    def __actions(self):
        return [i for i, elem in enumerate(self.children) if elem == None]