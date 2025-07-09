from collections import deque
import numpy as np

class Connect4():
    def __init__(self, other = None):
        if other == None:
            self.board = np.full([7, 6], 0, dtype = np.int8)
            self.descendants = { i: [None, 0] for i in range(7) }
            self.reward = 0
        else:
            self.board = other.board.copy()
            self.descendants = {action: [None, 0] for action, junk in other.descendants.items()}
            self.reward = other.reward

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([6, 7], "")
        dict = [" ", "X", "O"]
        for i in range(7):
            for j in range(6):
                board[-1 - j][i] = dict[self.board[i][j]]
        return str(board)

    def __update(self, action, pos):
        self.__descendants_function(action, pos)
        self.__reward_function(action, pos)

    def move(self, action, player):
        if self.descendants[action][0] != None:
            return self.descendants[action][0]
        temp = self.descendants[action][0] = Connect4(self)
        pos = np.count_nonzero(temp.board[action])
        temp.board[action][pos] = player
        temp.__update(action, pos)
        return temp

    def __descendants_function(self, action, pos):
        if pos == 5:
            del self.descendants[action]

    def __reward_function(self, action, pos):
        tokenizer = np.array([1, 4, 16]) # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 13: 3, 16: 13, 49: -3, 64: -13 } # converts the tokens into rewards throughout the board
        action = [action, pos]
        view = player = tokenizer[self.board[action[0]][action[1]]]
        
        self.reward = 0
        for step in [[0, 1], [1, -1], [1, 0], [1, 1]]:
            stack = deque()
            for sign in [-1, 1]:
                index = action
                limit = 3
                while (limit := limit - 1) >= 0 and (index := [index[0] + sign * step[0], index[1] + sign * step[1]]) and 0 <= index[0] < 7 and 0 <= index[1] < 6:
                    if sign == -1:
                        stack.appendleft(tokenizer[self.board[index[0]][index[1]]])
                        view += stack[-1]
                    else:
                        stack.append(tokenizer[self.board[index[0]][index[1]]])
                        view += stack[-1]
                        if len(stack) == 4:
                            view -= stack.popleft()
                            if view == 16 or view == 64:
                                self.descendants = {}
                                self.reward = parser[view]
                                return
                            self.reward += parser.get(view, 0) - parser.get(view - player + 1, 0)