# -*- coding: utf-8 -*-
"""
ImageData class to preprocess dicom and contour data
"""

import os
import glob
import numpy as np
import pandas as pd
from utils import parsing, image_processing
import traceback
import logging


# global variables for dicoms and contours folders
LINK_DICOM_FIELD = 'patient_id'
LINK_CONTOUR_FIELD = 'original_id'
ICONTOUR_SUBFOLDER = 'i-contours'
OCONTOUR_SUBFOLDER = 'o-contours'
ICONTOUR_FILENAME_FORMAT = 'IM-0001-{:04d}-icontour-manual.txt'
OCONTOUR_FILENAME_FORMAT = 'IM-0001-{:04d}-ocontour-manual.txt'
DELIM = '_'


class ImageData:
    def __init__(self, dicoms_path, contours_path, link_path, logger=None,
                 parse_icontour=True, parse_ocontour=True):
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
        :param parse_icontour: bool, whether to parse corresponding icontour file
        :param parse_ocontour: bool, whether to parse corresponding ocontour file
        """

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('dataset')
        else:
            self.logger = logger
        self.dicomn_contour_pairs = []
        self.dataset = []
        self.map_dicom_contour(dicoms_path, contours_path, link_path, parse_icontour, parse_ocontour)
        
    def map_dicom_contour(self, dicoms_path, contours_path, link_path,
                          parse_icontour=True, parse_ocontour=True):
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
                |         |----i-contours  
                |         |         |-----IM-0001-0001-icontour-manual.txt
                |         |         |-----IM-0001-0002-icontour-manual.txt
                |         |          ...
                |         |----o-contours  
                |         |         |-----IM-0001-0001-ocontour-manual.txt
                |         |         |-----IM-0001-0002-ocontour-manual.txt
                |         |         ...
                |------original_id_2
                |         |-----i-contours  
                |         |-----o-contours
                ...
        :param link_path: str, filepath to csv file that contains LINK_DICOM_FIELD and LINK_CONTOUR_FIELD columns.
                Each row in the csv file map a patient_id to original_id
        :param parse_icontour: bool, whether to parse corresponding icontour file
        :param parse_ocontour: bool, whether to parse corresponding ocontour file
        """

        # map patient_id to corresponding original_id
        self.logger.info('Reading link file at {} ...'.format(link_path))
        try:
            df_link = pd.read_csv(link_path)
            dict_link = dict(zip(df_link[LINK_DICOM_FIELD].values, df_link[LINK_CONTOUR_FIELD].values))
        except Exception:
            error_txt = traceback.format_exc()
            self.logger.error('Error reading link file {}. Trace back\n{}'.format(link_path, error_txt))
            return []

        # find all matching dicom-contour pairs
        self.logger.info('Mapping dicom and i-contour pairs...')
        self.dicomn_contour_pairs = []
        for patient_id in dict_link:
            img_path = os.path.join(dicoms_path, patient_id)
            icontour_path = os.path.join(contours_path, dict_link[patient_id], ICONTOUR_SUBFOLDER)
            ocontour_path = os.path.join(contours_path, dict_link[patient_id], OCONTOUR_SUBFOLDER)
            # patient's dicom and contour folder both exist
            if os.path.isdir(img_path) and \
                (not parse_icontour or os.path.isdir(icontour_path)) and \
                (not parse_ocontour or os.path.isdir(ocontour_path)):
                img_file_list = glob.glob(os.path.join(img_path, '[!._]*'))  # ignore ._ files created by Mac OS
                for img_file in img_file_list:
                    img_file_name = os.path.split(img_file)[-1]  # get dicom file name
                    img_number = os.path.splitext(img_file_name)[0]  # dicom file name without extension is image number
                    try:
                        img_number = int(img_number)
                    except ValueError:
                        self.logger.info('Patient {}: dicom filename {} does not match expected format.'
                                         .format(patient_id, img_file_name))

                    if parse_icontour:
                        icontour_filename = ICONTOUR_FILENAME_FORMAT.format(int(img_number))  # find matching i-contour file
                        icontour_file = os.path.join(icontour_path, icontour_filename)
                    else:
                        icontour_file = None
                        
                    if parse_ocontour:
                        ocontour_filename = OCONTOUR_FILENAME_FORMAT.format(int(img_number))  # find matching o-contour file
                        ocontour_file = os.path.join(ocontour_path, ocontour_filename)
                    else:
                        ocontour_file = None
                    if (not parse_icontour or os.path.isfile(icontour_file)) and \
                        (not parse_ocontour or os.path.isfile(ocontour_file)):
                        self.dicomn_contour_pairs.append((patient_id, img_file, icontour_file, ocontour_file))
                    else:
                        self.logger.info(
                            'Patient {}: i-contour or o-contour file for {} is missing.'\
                                        .format(patient_id, img_file_name))
            else:
                self.logger.info('{} or {} is missing.'.format(patient_id, dict_link[patient_id]))
        self.logger.info('Found {} matching dicom and contour paris.'.format(str(len(self.dicomn_contour_pairs))))

        return self.dicomn_contour_pairs

    def parse_files(self):
        """Parse dicom and corresponding contour file and add the processed data (img_id, img_data, imask, omask) to dataset.
            img_id is str with format patientID_imageNumber for identification purpose
            img_data is ndarray that contains 16-bit gray scale pixel data from dicom_file
            imask is ndarray that has the same shape as img_data and contains boolean mask of i-contour
            omask is ndarray that has the same shape as img_data and contains boolean mask of o-contour
        """

        self.logger.info('Parsing dicom and i-contour pairs...')
        for (patient_id, dicom_file, icontour_file, ocontour_file) in self.dicomn_contour_pairs:
            try:
                img_id = patient_id + DELIM + os.path.split(dicom_file)[-1]
                img_data = parsing.parse_dicom_file(dicom_file)[parsing.PIXEL_FIELD]
                
                if icontour_file is not None:
                    icountour_data = parsing.parse_contour_file(icontour_file)
                    imask = parsing.poly_to_mask(icountour_data, img_data.shape[1], img_data.shape[0])
                else:
                    imask = None
                    
                if ocontour_file is not None:
                    ocountour_data = parsing.parse_contour_file(ocontour_file)
                    omask = parsing.poly_to_mask(ocountour_data, img_data.shape[1], img_data.shape[0])
                else:
                    omask = None
                    
                self.dataset.append((img_id, img_data, imask, omask))
            except Exception:
                error_txt = traceback.format_exc()
                self.logger.error('Error parsing data for {}. Trace back\n {}'.format(dicom_file, error_txt))

        self.logger.info('Parsed {} dicom and contour pairs.'.format(str(len(self.dataset))))

    def save_result(self, output_path, normalize=False):
        """Save image data and mask together for visualization/inspection purpose
        :param output_path: str, folder path to store result
        :param normalize: bool, whether to normalize images to 0-255 scale.
        """

        self.logger.info('Saving dicom and contour images...')
        for (img_id, img_data, imask, omask) in self.dataset:
            save_path = os.path.join(output_path, img_id + '.png')
            img_list = [img for img in [img_data, imask, omask] if img is not None]
            image_processing.save_images(img_list, save_path, normalize=normalize)

        self.logger.info('Parsed result saved to {}.'.format(output_path))


def data_generator(datasource, batch_size=1, shuffle=True, augmentation=False, infinite_loop=True, logger=None):
    """A generator that returns img_data and corresponding mask.
    :param datasource:list of tuples (img_id, img_data, imask, omask)
    :param batch_size: int, number of observations to load in each batch
    :param shuffle: boolean, whether to shuffle the data at the begining of each epoch
    :param logger: logger object, for debug purpose
    """

    if len(datasource) == 0:
        raise ValueError('Datasource is empty.')

    i = 0
    index_list = np.arange(len(datasource))
    if shuffle:
        np.random.shuffle(index_list)

    flag = True
    while flag:  # loop over the datasource
        batch_img_id = []
        batch_img = []
        batch_mask = []
        for b in range(batch_size):            
            (img_id, img_data, mask, _) = datasource[index_list[i]]
            img_data = img_data.astype('float')/np.max(img_data)
            if augmentation:
                (img_data, mask) = image_processing.augment_image_pair(img_data, mask)
            
            batch_img_id.append(img_id)
            batch_img.append(np.stack((img_data,)*3, axis=-1))
            batch_mask.append(mask)

            i += 1
            if i == len(index_list):  # move back to the first index if reach the end of index_list
                if infinite_loop: # reset index to loop over the dataset again
                    i = 0
                else: # stop looping over the datasource
                    flag = False
                    break

        batch_img = np.array(batch_img)
        batch_mask = np.expand_dims(np.array(batch_mask), axis=3)
        
        if logger is not None:
            logger.info('Fetching data for {}, batch img shape {}, mask shape {}'.format(str(batch_img_id),
                        str(batch_img.shape), str(batch_mask.shape)))

        yield (batch_img, batch_mask)