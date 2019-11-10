# -*- coding: utf-8 -*-
"""
Pipeline to parse Dicom and contour files
"""

import os
from utils import dataset
import argparse
import logging


if __name__ == '__main__':
    # parse parameters
    parser = argparse.ArgumentParser(description='Pipeline to parse dicom and contour files')
    parser.add_argument('dicoms_path', help='A folderpath for dicom images')
    parser.add_argument('contours_path', help='A folderpath for contour images')
    parser.add_argument('link_path', help='A filepath to csv that link dicom and contour')
    parser.add_argument('--output_path', help='A folderpath to store visualization result')
    parser.add_argument('--parse_icontour', action='store_true', help='Whether to parse i-contour files')
    parser.add_argument('--parse_ocontour', action='store_true', help='Whether to parse o-contour files')
    args = parser.parse_args()

    # Initiate logger
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler('pipeline_dicom_contour.log'),
                                                      logging.StreamHandler()])
    logger = logging.getLogger('pipeline_dicom_contour')

    # Intialize ImageData
    image_data = dataset.ImageData(dicoms_path=args.dicoms_path, contours_path=args.contours_path,
                                   link_path=args.link_path, parse_icontour=args.parse_icontour,
                                   parse_ocontour=args.parse_ocontour, logger=logger)

    # Parse dicom and contour files
    image_data.parse_files()

    # Save parsed result
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
        image_data.save_result(args.output_path, normalize=False)
