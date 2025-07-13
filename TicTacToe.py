import numpy as np
from enum import Enum

class GameStatus(Enum):
    X_WINS = 1
    O_WINS = 2
    TIE = 0
    ONGOING = -1

class TicTacToe():
    table = [" ", "X", "O"]
    directions = np.array([[0, 1, 2], [0, 3, 6], [0, 2, 4], [0, 4, 8]]) # 4 different checks: horizontal, vertical, top right lower left, top left lower right
    
    def __init__(self, other = None):
        if isinstance(other, TicTacToe):
            self.board = other.board.copy()
        elif isinstance(other, int):
            self.board = np.empty([9], dtype = np.int8)
            for i in range(9):
                self.board[i] = other % 3
                other //= 3
        else:
            self.board = np.full([9], 0, dtype = np.int8)

        self.state_map = {}
        self.hash = None
        self.game_status = None

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = TicTacToe.table[elem]
        return str(board)
    
    @property
    def all_valid_actions(self):
        return [i for i, elem in enumerate(self.board) if elem == 0]

    """This method returns all unique actions by simulating each valid move, reducing symmetric board states, and storing only distinct outcomes in state_map. It avoids redundant exploration by hashing reduced boards and skipping duplicates. The result is a set of actions that lead to unique board configurations."""
    def get_actions(self, player):
        if not self.state_map:
            next_states = set()
            for action in self.all_valid_actions:
                board = TicTacToe(self)
                board.move(action, player)
                board.symmetry_reduction()
                if hash(board) not in next_states:
                    next_states.add(hash(board))
                    self.state_map[action] = board
        return list(self.state_map.keys())

    """This method either applies a move directly to the board if a player is given, or loads a precomputed symmetric board from state_map if not. It resets the cached hash in both cases. When loading from state_map, it also evaluates the resulting game state."""
    def move(self, action, player, apply_symmetry_reduction = False):
        state_map, self.state_map = self.state_map, {}
        self.game_status = None
        if apply_symmetry_reduction and state_map:
            self.board = state_map[action].board
            self.hash = state_map[action].hash
            self.game_status = state_map[action].game_status
        else:
            self.board[action] = player
            self.hash = None
            if apply_symmetry_reduction:
                self.__evaluate(action)
                self.symmetry_reduction()
        return self.__evaluate(action)

    # idk how this works; convoluted as shit; don't fuck with it
    """This method determines the game status after a move by checking if the current player formed a line of three. It calculates possible winning lines based on the move's position and compares the board values. If no win is found and no actions remain, it returns a tie; otherwise, the game continues."""
    def __evaluate(self, action, is_symmetry_reduced = False):
        def foo():
            if is_symmetry_reduced and action != 4:
                bases = [3, 1, 2, 0, 0, 6, 0, 2]
                directions = TicTacToe.directions[[0, 1, 2, 3, 0, 0, 1, 1]]
                if action % 2:
                    bases[2:4] = [None, None]
                else:
                    bases[0:2] = [None, None]
            else:
                bases = [(action // 3) * 3, action % 3, 2 if action in [2, 4, 6] else None, 0 if action in [0, 4, 8] else None]
                directions = TicTacToe.directions
            for base, direction in zip(bases, directions):
                if base is not None:
                    temp = direction + base
                    if 0 != self.board[temp[0]] == self.board[temp[1]] == self.board[temp[2]]:
                        return GameStatus.X_WINS if self.board[base] == 1 else GameStatus.O_WINS
                        
            if self.all_valid_actions == []:
                return GameStatus.TIE
            return GameStatus.ONGOING
        
        if not self.game_status:
            self.game_status = foo()
        return self.game_status
    
    """This __hash__ method encodes the Tic-Tac-Toe board as a unique base-3 integer by treating each cell as a digit. It caches the result in self.hash to avoid redundant computation. This allows fast comparisons and dictionary lookups for board states."""
    def __hash__(self):
        if self.hash is None:
            self.hash = sum(int(cell) * (3 ** i) for i, cell in enumerate(self.board))
        return self.hash

    """This function finds the canonical (lowest hash) representation of a Tic-Tac-Toe board by applying all 7 geometric symmetries (flips and rotations). It tests each transformed version of the board, keeps the one with the smallest hash, and sets the current board to that configuration. This helps reduce duplicate symmetric states in reinforcement learning or search."""
    def symmetry_reduction(self):
        board_copy = self.board.copy()
        lowest_hash, best_board = hash(self), board_copy
        #                 horizontal flip,       vertical flip,         rotate ccw,            rotate cw,             rotate 180,              diagonal flip,       anti-diagonal flip
        for transform in [lambda r, c: (r, 2-c), lambda r, c: (2-r, c), lambda r, c: (c, 2-r), lambda r, c: (2-c, r), lambda r, c: (2-r, 2-c), lambda r, c: (c, r), lambda r, c: (2-c, 2-r)]:
            i = 0
            for r in range(3):
                for c in range(3):
                    new_r, new_c = transform(r, c)
                    new_i = new_r * 3 + new_c
                    self.board[new_i] = board_copy[i]
                    i += 1

            self.hash = None
            if hash(self) < lowest_hash:
                lowest_hash, best_board = hash(self), self.board.copy()

        self.hash, self.board = lowest_hash, best_board
