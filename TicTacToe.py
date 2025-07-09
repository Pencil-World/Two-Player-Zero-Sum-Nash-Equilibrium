import numpy as np

class TicTacToe():
    dict = [" ", "X", "O"]
    
    def __init__(self, other = None):
        if other == None:
            self.board = np.full([9], 0, dtype = np.int8)
        else:
            self.board = other.board.copy()

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = TicTacToe.dict[elem]
        return str(board)

    def __update(self, action):
        self.__descendants_function(action)
        self.__reward_function(action)

    @property
    def get_actions(self):
        return [i for i, elem in enumerate(self.board) if elem == 0]

    def move(self, action, player):
        if self.descendants[action][0] != None:
            return self.descendants[action][0]
        temp = self.descendants[action][0] = TicTacToe(self)
        temp.board[action] = player
        temp.__update(action)
        return temp

    def __evaluate(self, action):
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            if 0 != self.board[coord] == self.board[coord + change] == self.board[coord + change + change]:
                self.progress = "RED"
                return
        self.progress = "GREEN"
