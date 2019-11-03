"""Parsing code for DICOMS and contour files"""

import os
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
from PIL import Image, ImageDraw


# global variable for dicom file format
INTERCEPT_FIELD = 'RescaleIntercept'
SLOPE_FIELD = 'RescaleSlope'
PIXEL_FIELD = 'pixel_data'


def parse_contour_file(filename):
    """Parse the given contour filename

    :param filename: filepath to the contourfile to parse
    :return: list of tuples holding x, y coordinates of the contour
    """

    if not os.path.isfile(filename):
        raise ValueError(filename + ' does not exist.')

    coords_lst = []
    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()
            try:
                x_coord = float(coords[0])
                y_coord = float(coords[1])
            except ValueError:
                raise ValueError('Some coordinates are not numeric.')
            coords_lst.append((x_coord, y_coord))
    return coords_lst


def parse_dicom_file(filename):
    """Parse the given DICOM filename

    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    if not os.path.isfile(filename):
        raise ValueError(filename + ' does not exist.')

    try:
        dcm = pydicom.read_file(filename)
    except InvalidDicomError:
        raise ValueError(filename + ' is not a valid DCOM file.')

    dcm_image = dcm.pixel_array
    if INTERCEPT_FIELD in dcm and SLOPE_FIELD in dcm:
        try:
            slope = float(dcm.SLOPE_FIELD)
            intercept = float(dcm.INTERCEPT_FIELD)
        except ValueError:
            raise ValueError('Intercept or slope is not numeric.')
        dcm_image = dcm_image * slope + intercept

    dcm_dict = {PIXEL_FIELD: dcm_image}
    return dcm_dict


def poly_to_mask(polygon, width, height):
    """Convert polygon to mask

    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # Reference http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask
