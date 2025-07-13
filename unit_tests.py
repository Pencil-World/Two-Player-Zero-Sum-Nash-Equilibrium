import unittest
from TicTacToe import TicTacToe, GameStatus
import numpy as np
from train_model import *
from model_manager import *

class TestTicTacToe(unittest.TestCase):
    def test_get_actions(self):
        return
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

class TestRLLoop(unittest.TestCase):
    def test_best_move(self):
        QTable = {
            0: [[1, 3, 0, 0, 0, 0, 0, 0, 0],    np.array([])],
            1: [[0, 0, 0, 0, 0, 0, 0, 0],       np.array([100, 100, 100, 10, 100, 100, 100, 100, 100])],
            3: [None,                           np.array([100, -1, 100, 100, 100, 100, 3, 100, 100])]
        }

        action = best_action(QTable, QTable[0][0], 1)
        self.assertEqual(action, 1)

    def test_generate_episode(self):
        pass

    def test_td_lambda(self):
        return
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

if __name__ == '__main__':
    unittest.main()
