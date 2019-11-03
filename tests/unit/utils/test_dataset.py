# -*- coding: utf-8 -*-
"""
Unittest for utils/dataset.py

"""

import unittest
from utils import dataset
import numpy as np


class TestDataset(unittest.TestCase):
    def test_data_generator(self):
        np.random.seed(13) # fix seed so that the randomization is fixed        
        datasource = [('a', 10, 1), ('b', 20, 2), ('c', 30, 3), ('d', 40, 4)] # create a datasource

        # generate data
        generated_data = []
        batch_size = 2
        for i in range(2):
            data_gen = dataset.data_generator(datasource, batch_size=batch_size, shuffle=True)
            for j in range(len(datasource)//batch_size):
                generated_data.append(data_gen.__next__())
                
        expected = [([20, 40], [2, 4]), ([10, 30], [1, 3]), ([20, 40], [2, 4]), ([30, 10], [3, 1])]
        
        np.testing.assert_equal(generated_data, expected)
        
    def test_data_generator_loopback(self):
        np.random.seed(13) # fix seed so that the randomization is fixed        
        datasource = [('a', 10, 1), ('b', 20, 2), ('c', 30, 3), ('d', 40, 4)] # create a datasource

        # generate data
        generated_data = []
        batch_size = 3
        for i in range(1):
            data_gen = dataset.data_generator(datasource, batch_size=batch_size, shuffle=True)
            for j in range(len(datasource)//batch_size+1):
                generated_data.append(data_gen.__next__())
                
        expected = [([20, 40, 10], [2, 4, 1]), ([30, 20, 40], [3, 2, 4])]
        
        np.testing.assert_equal(generated_data, expected)


