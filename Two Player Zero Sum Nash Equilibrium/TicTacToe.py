import numpy as np

class TicTacToe():
    def __init__(self, board = []):
        self.board = np.full([9], 0, dtype = np.int8)
        if len(board) > 0:
            for i, elem in enumerate(board):
                self.board[i] = elem
        self.update()

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        dict = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = dict[elem]
        return str(board)

    def update(self, action = None):
        if action != None:
            self.board[action] = sum(self.board) % 3 + 1
        self.children = { i: [None, 0] for i, elem in enumerate(self.board) if not elem }
        self.reward = self.__reward
        self.actions = self.__actions

    def move(self, action):
        if self.children[action][0] != None:
            return self.board[action]
        temp = self.children[action][0] = TicTacToe(self.board)
        temp.update(action)
        return temp

    @property
    def __reward(self):
        tokenizer = np.array([1, 3, 9]) # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 7: 3, 9: 9, 19: -3, 27: -9 } # converts the tokens into rewards throughout the board
        reward = 0
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            temp = sum(tokenizer[self.board[[coord, coord + change, coord + change + change]]])
            if temp == 9 or temp == 27:
                self.children = {}
                return parser[temp]
            reward += parser.get(temp, 0)
        return reward

    @property
    def __actions(self):
        return list(self.children.keys())