# -*- coding: utf-8 -*-
"""
Unittest for utils/metrics.py

"""

import unittest
from utils import metrics
import numpy as np


class TestMetrics(unittest.TestCase):
    def test_iou_score(self):
        a = np.array([1, 0, 0, 1])
        b = np.array([1, 1, 0, 0])
        iou = metrics.iou_score(a, b)
        expected = 1/3
        np.testing.assert_almost_equal(iou, expected, decimal=3)

    def test_dice_score(self):
        a = np.array([1, 0, 0, 1])
        b = np.array([1, 1, 0, 0])
        iou = metrics.dice_score(a, b)
        expected = 0.5
        np.testing.assert_almost_equal(iou, expected, decimal=3)
