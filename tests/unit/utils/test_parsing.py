# -*- coding: utf-8 -*-
"""
Unittest for utils/parsing.py

"""

import unittest
import os
import pydicom.uid as uid
from pydicom.dataset import Dataset, FileDataset
from utils import parsing
import numpy as np
import tempfile


class TestParsing(unittest.TestCase):
    def test_parse_contour_file(self):
        # create a temporary contour file
        temp_dir = tempfile.gettempdir()  # get temp directory
        temp_file = os.path.join(temp_dir, 'contour.txt')
        with open(temp_file, 'w') as f:
            f.write('0.0  0.0\n')
            f.write('1.0  1.0\n')
            f.write('2.0  2.0\n')

        # parse file
        contours = parsing.parse_contour_file(temp_file)
        expected = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        os.remove(temp_file)

        np.testing.assert_equal(contours, expected)

    def test_parse_dicom_file(self):
        # create a temporary dicom file
        temp_dir = tempfile.gettempdir()  # get temp directory
        temp_file = os.path.join(temp_dir, 'dicom.dcm')

        # Reference https://stackoverflow.com/questions/14350675/create-pydicom-file-from-numpy-array
        pixel_array = np.diag([1, 2, 3]).astype(np.uint16)  # create a diagonal matrix
        file_meta = Dataset()
        ds = FileDataset(temp_file, {}, file_meta=file_meta, preamble=b'\x00' * 128)
        file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.Columns = pixel_array.shape[0]
        ds.Rows = pixel_array.shape[1]
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_array.tostring()
        ds.save_as(temp_file)

        # parse file
        im_data = parsing.parse_dicom_file(temp_file)
        os.remove(temp_file)

        np.testing.assert_equal(im_data, {parsing.PIXEL_FIELD: pixel_array})

    def test_poly_to_mask(self):
        # create input
        polygon = [(0, 0), (0, 2), (2, 2), (2, 0)]
        width = height = 3

        # poly_to_mask
        mask = parsing.poly_to_mask(polygon, width, height)
        expected = np.array([[False, False, False],
                             [False,  True, False],
                             [False, False, False]])

        np.testing.assert_equal(mask, expected)
