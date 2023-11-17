import numpy as np

class TicTacToe():
    def __init__(self, other = None):
        if other == None:
            self.board = np.full([9], 0, dtype = np.int8)
            self.children = { i: [None, 0] for i in range(9) }
            self.reward = 0
            self.actions = []
        else:
            self.board = other.board.copy()
            self.children = other.children.copy()
            # self.reward = other.reward
            self.actions = other.actions.copy()

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        dict = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = dict[elem]
        return str(board)

    def update(self, action):
        del self.children[action]
        self.reward = self.__reward(action)
        self.__actions(action)

    def move(self, action, player):
        if self.children[action][0] != None:
            return self.children[action][0]
        temp = self.children[action][0] = TicTacToe(self)
        temp.board[action] = player
        temp.update(action)
        return temp

    def __reward(self, action):
        tokenizer = np.array([1, 3, 9]) # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 7: 3, 9: 13, 19: -3, 27: -13 } # converts the tokens into rewards throughout the board
        reward = 0
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            temp = sum(tokenizer[self.board[[coord, coord + change, coord + change + change]]])
            if temp == 9 or temp == 27:
                self.children = {}
                return parser[temp]
            reward += parser.get(temp, 0)
        return reward

    def __actions(self, action):
        self.actions.remove(action)