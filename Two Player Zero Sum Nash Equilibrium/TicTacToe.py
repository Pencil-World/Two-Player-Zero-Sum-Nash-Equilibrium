import numpy as np

class TicTacToe():
    def __init__(self, other = None):
        if other == None:
            self.board = np.full([9], 0, dtype = np.int8)
            self.descendants = { i: [None, 0] for i in range(9) }
            self.reward = 0
        else:
            self.board = other.board.copy()
            self.descendants = {action: [None, 0] for action, junk in other.descendants.items()}
            # self.reward = other.reward

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        dict = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = dict[elem]
        return str(board)

    def __update(self, action):
        self.__descendants_function(action)
        self.__reward_function(action)

    def move(self, action, player):
        if self.descendants[action][0] != None:
            return self.descendants[action][0]
        temp = self.descendants[action][0] = TicTacToe(self)
        temp.board[action] = player
        temp.__update(action)
        return temp

    def __descendants_function(self, action):
        del self.descendants[action]

    def __reward_function(self, action):
        tokenizer = np.array([1, 3, 9]) # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 7: 3, 9: 13, 19: -3, 27: -13 } # converts the tokens into rewards throughout the board
        self.reward = 0
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            temp = sum(tokenizer[self.board[[coord, coord + change, coord + change + change]]])
            if temp == 9 or temp == 27:
                self.descendants = {}
                self.reward = parser[temp]
                return
            self.reward += parser.get(temp, 0)