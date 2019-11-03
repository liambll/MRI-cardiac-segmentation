# -*- coding: utf-8 -*-
"""
Pipeline to parse Dicom and contour files
"""

import os
from utils import dataset
import argparse

if __name__ == '__main__':
    dicoms_path = 'C:/SourceCode/dicom-code-challenge/dataset/final_data/dicoms'
    contours_path = 'C:/SourceCode/dicom-code-challenge/dataset/final_data/contourfiles'
    link_path = 'C:/SourceCode/dicom-code-challenge/dataset/final_data/link.csv'
    output_path = 'C:/SourceCode/dicom-code-challenge/dataset/final_data/parsed_dicoms_mask'

    # parse parameters
    parser = argparse.ArgumentParser(description='Pipeline to parse dicom and contour files')
    parser.add_argument('dicoms_path', help='A folderpath for dicom images')
    parser.add_argument('contours_path', help='A folderpath for contour images')
    parser.add_argument('link_path', type=int, help='A filepath to csv that link dicom and contour')
    parser.add_argument('--output_path', help='A folderpath to store visualization result')
    args = parser.parse_args()
        
    # Intialize ImageData
    image_data = dataset.ImageData(dicoms_path, contours_path, link_path)

    # Parse dicom and contour files
    image_data.parse_files()

    # Visualize parsed result
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        image_data.visualize_result(output_path)
