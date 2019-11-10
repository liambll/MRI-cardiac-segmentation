# -*- coding: utf-8 -*-
"""
Unittest for utils/image_processing.py

"""

import unittest
from utils import image_processing
import numpy as np


class TestImageProcessing(unittest.TestCase):
    def test_normalize_image(self):
        img = np.array([1, 2, 3, 4, 5])
        img_normalized = image_processing.normalize_image(img)
        expected = [51, 102, 153, 204, 255]

        np.testing.assert_equal(img_normalized, expected)

    def test_convex_image(self):
        img = np.array([[255, 255, 255, 255],
                        [0, 0, 0, 255],
                        [0, 0, 0, 255],
                        [255, 0, 0, 255]], dtype='uint8')
        img_convex = image_processing.convex_image(img)
        expected = [[255, 255, 255, 255],
                        [0, 255, 255, 255],
                        [0, 0, 255, 255],
                        [255, 0, 0, 255]]

        np.testing.assert_equal(img_convex, expected)
             
    def test_convex_image_largest_hull(self):
        img = np.array([[255, 255, 255, 255],
                        [0, 0, 0, 255],
                        [0, 0, 0, 255],
                        [255, 0, 0, 255]], dtype='uint8')
        img_convex = image_processing.convex_image(img, largest_hull=True)
        expected = [[255, 255, 255, 255],
                        [0, 255, 255, 255],
                        [0, 0, 255, 255],
                        [0, 0, 0, 255]]

        np.testing.assert_equal(img_convex, expected)
        
    def test_overlay_images(self):
        # Create a white background
        img = np.zeros((2, 2), dtype='uint8')
        
        # Create a mask with horizontal line at the 1st row
        mask1 = np.zeros((2, 2), dtype='uint8')
        mask1[:, 0] = 1
        
        # Create a mask with vertical line at the 1st column
        mask2 = np.zeros((2, 2), dtype='uint8')
        mask2[0, :] = 1
        
        
        result = image_processing.overlay_images(background=img, masks=[mask1, mask2],
                                                 colors=[(255, 0, 0), (0, 0, 255)])
        expected = [[[0, 0, 255], [0, 0, 255]],
                    [[255,   0,   0], [0, 0, 0]]]

        np.testing.assert_equal(result, expected)
        
