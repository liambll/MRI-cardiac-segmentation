# -*- coding: utf-8 -*-
"""
ImageData class to preprocess dicom and contour data
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from utils import parsing
import traceback
import logging


# global variables for dicoms and contours folders
LINK_DICOM_FIELD = 'patient_id'
LINK_CONTOUR_FIELD = 'original_id'
ICONTOUR_SUBFOLDER = 'i-contours'
ICONTOUR_FILENAME_FORMAT = 'IM-0001-{:04d}-icontour-manual.txt'
DELIM = '_'


class ImageData:
    def __init__(self, dicoms_path, contours_path, link_path, logger=None):
        """
        :param dicoms_path: str, folderpath contains patients' image. dicoms_path should have the below structure:
            dicoms_path
                |------patient_id_1
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                |------patient_id_2
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                ...
        :param contours_path: str, folderpath contains contour files. contours_path should have the below structure:
            contours_path
                |------original_id_1
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                |------original_id_2
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                ...
        :param link_path: str, filepath to csv file that contains LINK_DICOM_FIELD and LINK_CONTOUR_FIELD columns.
                Each row in the csv file map a patient_id to original_id
        :param logger: Logger object
        """

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('dataset')
        else:
            self.logger = logger
        self.dicomn_contour_pairs = self.map_dicom_contour(dicoms_path, contours_path, link_path)
        self.dataset = []

    def map_dicom_contour(self, dicoms_path, contours_path, link_path):
        """Map each dicom file to corresponding contour file, if available.
        :param dicoms_path: str, folderpath contains patients' image. dicoms_path should have the below structure:
            dicoms_path
                |------patient_id_1
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                |------patient_id_2
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                ...
        :param contours_path: str, folderpath contains contour files. contours_path should have the below structure:
            contours_path
                |------original_id_1
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                |------original_id_2
                |         |-----1.dcm
                |         |-----2.dcm
                |         ...
                ...
        :param link_path: str, filepath to csv file that contains LINK_DICOM_FIELD and LINK_CONTOUR_FIELD columns.
                Each row in the csv file map a patient_id to original_id
        """

        # map patient_id to corresponding original_id
        try:
            df_link = pd.read_csv(link_path)
            dict_link = dict(zip(df_link[LINK_DICOM_FIELD].values, df_link[LINK_CONTOUR_FIELD].values))
        except Exception:
            error_txt = traceback.format_exc()
            self.logger.error('Error reading link file {}. Trace back\n{}'.format(link_path, error_txt))
            return []

        # find all matching dicom-contour pairs
        dicomn_contour_pairs = []
        for patient_id in dict_link:
            img_path = os.path.join(dicoms_path, patient_id)
            label_path = os.path.join(contours_path, dict_link[patient_id], ICONTOUR_SUBFOLDER)
            if os.path.isdir(img_path) and os.path.isdir(label_path):  # patient's dicom and contour folder both exist
                img_file_list = glob.glob(os.path.join(img_path, '[!._]*'))  # ignore ._ files created by Mac OS
                for img_file in img_file_list:
                    img_file_name = os.path.split(img_file)[-1]  # get dicom file name
                    img_number = os.path.splitext(img_file_name)[0]  # dicom file name without extension is image number
                    try:
                        img_number = int(img_number)
                    except ValueError:
                        self.logger.info('Patient {}: dicom filename {} does not match expected format.'
                                         .format(patient_id, img_file_name))

                    contour_filename = ICONTOUR_FILENAME_FORMAT.format(int(img_number))  # find matching contour file
                    contour_file = os.path.join(label_path, contour_filename)
                    if os.path.isfile(contour_file):
                        dicomn_contour_pairs.append((patient_id, img_file, contour_file))
                    else:
                        self.logger.info(
                            'Patient {}: i-contour_file for {} is missing.'.format(patient_id, img_file_name))
            else:
                self.logger.info('{} or {} is missing.'.format(patient_id, dict_link[patient_id]))
        self.logger.info('Found {} matching dicom and i-contour pairs.'.format(str(len(dicomn_contour_pairs))))

        return dicomn_contour_pairs

    def parse_files(self):
        """Parse dicom and corresponding contour file and add the processed data (img_id, img_data, mask) to dataset.
            img_id is str with format patientID_imageNumber for identification purpose
            img_data is ndarray that contains 16-bit gray scale pixel data from dicom_file
            mask is ndarray that has the same shape as img_data and contains boolean mask of i-contour
        """

        for (patient_id, dicom_file, contour_file) in self.dicomn_contour_pairs:
            try:
                img_id = patient_id + DELIM + os.path.split(dicom_file)[-1]
                img_data = parsing.parse_dicom_file(dicom_file)[parsing.PIXEL_FIELD]
                countour_data = parsing.parse_contour_file(contour_file)
                mask = parsing.poly_to_mask(countour_data, img_data.shape[1], img_data.shape[0])
                self.dataset.append((img_id, img_data, mask))
            except Exception:
                error_txt = traceback.format_exc()
                self.logger.error('Error parsing ({}, {}). Trace back\n {}'.format(dicom_file, contour_file, error_txt))

        self.logger.info('Parsed {} dicom and i-contour pairs.'.format(str(len(self.dataset))))

    def visualize_result(self, output_path):
        """Save image data and mask together for visualization/inspection purpose
        :param output_path: str, folder path to store result
        """

        for (img_id, img_data, mask) in self.dataset:
            # normalize 16bit gray scale to 0-255
            img_data = img_data.astype('float') / np.max(img_data) * 255
            mask = mask.astype('float') * 255

            # put img_data and mask side by side
            result = np.concatenate([img_data, mask], axis=1).astype('uint8')

            # save result
            im = Image.fromarray(result, 'L')
            output_filepath = os.path.join(output_path, img_id + '.png')
            im.save(output_filepath)

        self.logger.info('Visualization result saved to {}.'.format(output_path))


def data_generator(datasource, batch_size=1, shuffle=True, logger=None):
    """A generator that returns img_data and corresponding mask.
    :param datasource:list of tuples (img_id, img_data, mask)
    :param batch_size: int, number of observations to load in each batch
    :param shuffle: boolean, whether to shuffle the data at the begining of each epoch
    :param logger: logger object
    """

    i = 0
    index_list = np.arange(len(datasource))
    if shuffle:
        np.random.shuffle(index_list)

    while True:  # loop over the datasource indefinitely
        batch_img_id = []
        batch_img = []
        batch_mask = []
        for b in range(batch_size):
            (img_id, img_data, mask) = datasource[index_list[i]]
            batch_img_id.append(img_id)
            batch_img.append(img_data)
            batch_mask.append(mask)

            i += 1
            if i == len(index_list):  # move back to the first index if reach the end of index_list
                i = 0

        batch_img = np.array(batch_img)
        batch_mask = np.array(batch_mask)
        
        if logger is not None:
            logger.info('Fetching data for {}.'.format(str(batch_img_id)))

        yield (batch_img, batch_mask)