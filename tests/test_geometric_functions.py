import unittest
import numpy as np
from giwerm.geometric_functions import alpha_geodesic


class TestGeometricFunctions(unittest.TestCase):
    def test_alpha_geodesic(self):
        case_1 = {"alpha": 1, "lmd": 0.5,
                  "a": np.array([0, 0]), "b": np.array([1, 1])}

        result_1 = alpha_geodesic(case_1["a"], case_1["b"], case_1["lmd"], case_1["alpha"])
        print(result_1)

        self.assertTrue(np.array_equal(result_1, np.array([0.5, 0.5])))
