from collections import deque
import numpy as np

class Connect4():
    def __init__(self, other = None):
        if other == None:
            self.board = np.full([6, 7], 0, dtype = np.int8)
            self.descendants = { i: [None, 0] for i in range(9) }
            self.reward = 0
        else:
            self.board = other.board.copy()
            self.descendants = other.descendants.copy()
            self.reward = other.reward

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([6, 7], "")
        dict = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 6][i % 7] = dict[elem]
        return str(board)

    def __update(self, action):
        self.__descendants_function(action)
        self.__reward_function(action)

    def move(self, action, player):
        if self.descendants[action][0] != None:
            return self.descendants[action][0]
        temp = self.descendants[action][0] = Connect4(self.board)
        temp.board[action[0]][action[1]] = player
        temp.__update(action)
        return temp

    def __descendants_function(self, action):
        if self.board[action].count_nonzero() == len(self.board[action]):
            del self.descendants[action]

    # read board from left to right, up to down
    def __reward_function(self, action):
        tokenizer = np.array([1, 3, 9]) # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 13: 3, 12: 13, 28: -3, 36: -13 } # converts the tokens into rewards throughout the board
        player = self.board[action[0]][action[1]]
        view = 0
        for step in [[0, 1], [1, 0], [1, -1], [1, 1]]:
            stack = deque()
            limit = 3
            for sign in [-1, 1]:
                index = action.copy()
                while (limit := limit - 1) >= 0 and 0 <= index[0] < 6 and 0 <= index[1] < 7:
                    if sign == -1:
                        stack.append(tokenizer[self.board[index[0]][index[1]]])
                    else:
                        stack.appendleft(tokenizer[self.board[index[0]][index[1]]])
                        if len(stack) > 4:
                            view += stack[0] - stack.pop()
                            if view == 12 or view == 36:
                                self.descendants = {}
                                self.reward = parser[view]
                                return
                            self.reward += parser.get(view, 0) - parser.get(view + 1 - player, 0)
                    index = [index[0] + sign * step[0], index[1] + sign * step[1]]
                limit = 4