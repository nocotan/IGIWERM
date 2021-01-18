import unittest
import numpy as np

from giwerm.model_selection import train_test_split


class TrainTestSplitTest(unittest.TestCase):
    def test_shape(self):
        x = np.random.rand(1000, 32)
        y = np.random.rand(1000)

        train_x, train_y, test_x, test_y = train_test_split(x, y)

        self.assertEqual(len(train_x), len(train_y))
        self.assertEqual(len(test_x), len(test_y))
