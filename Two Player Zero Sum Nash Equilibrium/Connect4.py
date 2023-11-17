import numpy as np

class Connect4():
    def __init__(self, board = []):
        self.board = np.full([6, 7], 0, dtype = np.int8)
        if len(board) > 0:
            for i, elem in enumerate(board):
                self.board[i] = elem
        else:
            self.player = 2
        self.update()

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([6, 7], "")
        dict = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 6][i % 7] = dict[elem]
        return str(board)

    def update(self, action = None):
        if action != None:
            for index, elem in enumerate(self.board[action]):
                if elem == 0:
                    self.board[action][index] = self.player
        self.children = { i: [None, 0] for i, elem in enumerate(self.board) if not elem[-1] }
        self.reward = self.__reward
        self.actions = self.__actions

    def move(self, action):
        if self.children[action][0] != None:
            return self.children[action][0]
        temp = self.children[action][0] = Connect4(self.board)
        temp.player = [0, 2, 1][self.player]
        temp.update(action)
        return temp

    @property
    def __reward(self, action):
        tokenizer = np.array([1, 3, 9]) # tokenizes the elements " ", "X", and "O" into numerical forms
        parser = { 10: 1, 12: 10, 28: -1, 36: -10 } # converts the tokens into rewards throughout the board
        temp = 0
        for step in [[]]:
            
        for (origin, delta, repeat) in zip([[0, 0], [0, 0], [0, 2], [0, 3], [1, 0], [1, 5]], [[0, 1], [1, 0], [0, -1], [0, 1], [1, 0], [1, 0]], [6, 7, 2, 2, 2, 2]):
            for i in range(repeat):
                coord = [origin[0] + i * delta[0], origin[1] + i * delta[1]]
                for change, maximum_stuff in zip([[0, 0], [0, 0]], [[0, 1], [1, 0], [1, 1], []]):
                    counter = 0
                    streak = None
                    for junk in range(maximum_stuff):
                        if self.board[0][0] == streak:
                            counter += tokenizer[streak]
                        else:
                            if counter == 12 or counter == 36:
                                self.children = {}
                                return parser[counter]
                            reward += parser.get(counter, 0)
        return reward

    def __actions(self, action):
        return list(self.children.keys())