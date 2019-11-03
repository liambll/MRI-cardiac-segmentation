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
    parser.add_argument('--nb_epoch', type=int, default=5, help='Number of epoch to train model')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of observation in each batch')
    args = parser.parse_args()

    # Initiate logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('pipeline_dicom_contour')

    # Intialize ImageData
    image_data = dataset.ImageData(dicoms_path=args.dicoms_path, contours_path=args.contours_path,
                                   link_path=args.link_path, logger=logger)

    # Parse dicom and contour files
    image_data.parse_files()

    # Visualize parsed result
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
        image_data.visualize_result(args.output_path)

    # Perform model training
    # TODO: set up model and the below code portion will be part of train() function of the model
    for epoch in range(args.nb_epoch):
        logger.info('Epoch {}:'.format(str(epoch)))
        data_gen = dataset.data_generator(datasource=image_data.dataset, batch_size=args.batch_size, shuffle=True,
                                          logger=logger)
        nb_steps = len(image_data.dataset) // args.batch_size
        for step in range(nb_steps):
            batch_img, batch_mask = data_gen.__next__()
            # TODO: train step

        # TODO: val step
