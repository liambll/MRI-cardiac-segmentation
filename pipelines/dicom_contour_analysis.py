# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:22:36 2019

@author: liam.bui

Analysis to check data correctness and Perform some image analysis to segment out i-contour

"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import image_processing, metrics
import argparse
import logging


def read_parsed_images(data_path, img_dim=256):
    """Read pixel data and masks from parsed data saved by pipeline_dicom_contour.
    :param data_path:str, folder path that contain parsed dicom-contour images
        Each parsed dicom-contour image must have shape (img_dim, img_dim*3)
    :param img_dim: width of dicomn pixel data
    :return parsed_data: list of tuple (img_filename, img_data, imask, omask)
    """
    
    img_list = os.listdir(data_path)
    parsed_data = []
    for img_filename in img_list:
        img = cv2.imread(os.path.join(data_path, img_filename), -1) # -1 flag: read raw data image without any conversion
        img_data = img[:, :img_dim] # dicom pixel data
        imask = img[:, img_dim:2*img_dim] # i-contour mask
        omask = img[:, 2*img_dim:] # o-contour mask
        parsed_data.append((img_filename, img_data, imask, omask))
        
    return parsed_data
            
            
def overlay_contours(parsed_data, overlay_path):
    """Draw boundary of masks for all images
    :param parsed_data: list of tuple (img_filename, img_data, imask, omask)
    :param overlay_path: str, path to store images with mask boundaries overlayed
    """    
    
    for (img_filename, img_data, imask, omask) in parsed_data:
        # Overlay with red for i-contour and blue for o-contour
        overlay = image_processing.overlay_images(img_data, [imask, omask], colors=[(0, 0, 255), (255, 0, 0)])
        save_path = os.path.join(overlay_path, img_filename)
        image_processing.save_images([overlay], save_path, normalize=False)
        

def extract_pixel_intensity(parsed_data, normalized=False):
    """Extract intensity of of blood pool (inside i-contour) and heart muscle (outside i-contour but inside o-contour)
    :param parsed_data: list of tuple (img_filename, img_data, imask, omask)
    :param normalized: whether to normalize data in each image to 0-255 range
    :return list_blood_pool: list, contains pixel intensity of blood pool area in each image
    :return list_heart_muscle: list, contains pixel intensity of heart muscle area in each image
    """   

    list_filename = []
    list_blood_pool = []
    list_heart_muscle = []
    
    for (img_filename, img_data, imask, omask) in parsed_data:
        pixel_blood_pool = img_data[np.logical_and(omask > 0, imask > 0)] # inside i-contour and o-contour
        pixel_heart_muscle = img_data[np.logical_and(omask > 0, imask == 0)] # outside i-contour and inside o-contour
        if normalized:
            pixel_blood_pool = image_processing.normalize_image(pixel_blood_pool)
            pixel_heart_muscle = image_processing.normalize_image(pixel_heart_muscle)
        
        list_filename.append(img_filename.split('.')[0])
        list_blood_pool.append(pixel_blood_pool)
        list_heart_muscle.append(pixel_heart_muscle)
        
    return list_filename, list_blood_pool, list_heart_muscle     


def visualize_boxplot(list_blood_pool, list_heart_muscle, list_label, list_color=['r', 'b'], save_path=None):
    """Boxplot visualization to compare pixel intensity of blood pool and hear muscle
    :param list_blood_pool: list of data, each data is a list of pixel intensity of blood pool in an image
    :param list_heart_muscle: list of data, each data is a list pixel intensity of heart muscle in an image
    :param list_label: list, contains label for each image
    :param list_color: list of colors to plot for blood pool and hear muscle
    :param save_path: str, filepath to save the plot
    """
    
    fig, ax = plt.subplots()
    c = list_color[0]
    bp1 = plt.boxplot(list_blood_pool, boxprops=dict(color=c), medianprops=dict(color=c), capprops=dict(color=c),
                whiskerprops=dict(color=c), showfliers=False)
    c = list_color[1]
    bp2 = plt.boxplot(list_heart_muscle, labels=list_filename, boxprops=dict(color=c), medianprops=dict(color=c),
                      capprops=dict(color=c), whiskerprops=dict(color=c), showfliers=False)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Blood Pool', 'Heart Muscle'], loc='upper right')
    plt.title('Boxplot of Pixel intensity in Blood pool vs Heart muscle')
    plt.xlabel('Image')
    plt.ylabel('Pixel intensity')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # save plot
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        
def perform_otsu_thresholding(parsed_data):
    """Perform otsu thresholding for segmentation
    :param parsed_data: list of tuple (img_filename, img_data, imask, omask)
    :param post_processing: whether to perform convex hull postprocessing
    :return list_otsu_result: list, contains segmentation result for each image using otsu thresholding
    :return list_otsu_hull_result: list, contains segmentation result for each image using otsu thresholding
                                        and convex hull postprocessing
    """    
      
    list_otsu_result = []
    list_otsu_hull_result = []
    for (img_filename, img_data, imask, omask) in parsed_data:
        # Get only pixels inside ocontour and perform otsu thresholding
        pixel_ocontour = image_processing.normalize_image(img_data[omask > 0])
        thres, thres_img = cv2.threshold(pixel_ocontour, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        otsu_result = np.zeros(img_data.shape, dtype='uint8')
        otsu_result[omask > 0] = thres_img.flatten()
        
        # Get the largest convex hull on Otsu result
        otsu_hull_result = image_processing.convex_image(otsu_result, largest_hull=True)
        
        list_otsu_result.append(otsu_result)
        list_otsu_hull_result.append(otsu_hull_result)
        
    return list_otsu_result, list_otsu_hull_result
        

def evaluate_segmentation(parsed_data, prediction, save_path=None):
    """Perform evaluation of segmentation result
    :param parsed_data: list of tuple (img_filename, img_data, imask, omask)
    :param prediction: list, contains predicted i-contour mask
    :return list_iou_score: list, contains Intersection-Over-Union score for each image
    :return list_dice_score: list, contains Dice score for each image
    """
    
    list_iou_score = []
    list_dice_score = []
    for i in range(len(parsed_data)):
        (img_filename, img_data, imask, omask) = parsed_data[i]
        img_pred = prediction[i]/255
        list_iou_score.append(metrics.iou_score(imask, img_pred))
        list_dice_score.append(metrics.dice_score(imask, img_pred))
        
        if save_path:
            overlay = image_processing.overlay_images(img_data, [imask, img_pred], colors=[(0, 0, 255), (0, 255, 255)])
            image_processing.save_images([overlay], os.path.join(save_path, img_filename), normalize=False)
            
    return list_iou_score, list_dice_score

    
if __name__ == '__main__':
    data_path = 'C:/SourceCode/dicom-code-challenge/dataset/final_data/parsed_dicoms_mask/'
    analysis_path = 'C:/SourceCode/dicom-code-challenge/dataset/final_data/analysis/'
    
    # parse parameters
    parser = argparse.ArgumentParser(description='Analysis of dicom and contour files')
    parser.add_argument('data_path', help='A folderpath for parsed dicom-contour images')
    parser.add_argument('analysis_path', help='A folderpath to store analysis result')
    args = parser.parse_args()

    # Initiate logger
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler('dicom_contour_analysis.log'),
                                                      logging.StreamHandler()])
    logger = logging.getLogger('dicom_contour_analysis')


    # Read parsed data
    parsed_data = read_parsed_images(data_path, img_dim=256)
    
    ############################
    ## Check data correctness ##
    ############################
    # Visualize parsed result
    logger.info('Overlaying contours on dicom pixel data ...')
    overlay_path = os.path.join(analysis_path, 'overlay')
    os.makedirs(overlay_path, exist_ok=True)
    overlay_contours(parsed_data, overlay_path)
    logger.info('Overlayed result is saved to {}.\n'.format(overlay_path))
    
    # Exclude 'SCD0000501' for subsequent analysis because the data seems incorrect
    logger.info('Exclude all SCD0000501 images from subsequent analsysis.')
    parsed_data = [data for data in parsed_data if 'SCD0000501' not in data[0]]
        
    
    ##############################################
    ## Check feasibility of simple thresholding ##
    ##############################################
    # Box-Plot of pixel density and normalized pixel intensity for each image
    logger.info('Visualizing boxplot pixel intensity of  blood pool and hear muscle ...')
    boxplot_path = os.path.join(analysis_path, 'boxplot')
    os.makedirs(boxplot_path, exist_ok=True)
    
    list_filename, list_blood_pool, list_heart_muscle = extract_pixel_intensity(parsed_data, normalized=False)
    save_path = os.path.join(boxplot_path, 'pixel_intensity.png')
    visualize_boxplot(list_blood_pool=list_blood_pool, list_heart_muscle=list_heart_muscle,
                      list_label=list_filename, list_color=['r', 'b'], save_path=save_path)

    list_filename, list_blood_pool, list_heart_muscle = extract_pixel_intensity(parsed_data, normalized=True)
    save_path = os.path.join(boxplot_path, 'pixel_intensity_normalized.png')
    visualize_boxplot(list_blood_pool=list_blood_pool, list_heart_muscle=list_heart_muscle,
                      list_label=list_filename, list_color=['r', 'b'], save_path=save_path)   
    logger.info('Boxplot visualization is saved to {}.\n'.format(boxplot_path))
    

    ############################################
    ## Check feasibility of otsu thresholding ##
    ############################################
    # Perform Otsu Threshold
    logger.info('Perform Otsu thresholding ...')
    list_otsu_result, list_otsu_hull_result = perform_otsu_thresholding(parsed_data)

    # Evaluate segmentation result for Otstu thresholding
    otsu_path = os.path.join(analysis_path, 'otsu')
    os.makedirs(otsu_path, exist_ok=True)
    logger.info('Evaluating segmentation result with Otsu thresholding ...')
    otsu_result_path = os.path.join(analysis_path, 'otsu')
    os.makedirs(otsu_result_path, exist_ok=True)
    list_iou_score, list_dice_score = evaluate_segmentation(parsed_data=parsed_data, prediction=list_otsu_result,
                                           save_path=otsu_result_path)
    logger.info('--Mean IoU score: {}, Dice score: {}\n'.format(str(np.mean(list_iou_score)),
                str(np.mean(list_dice_score))))
        
    # Evaluate segmentation result for Otstu thresholding
    logger.info('Evaluating segmentation result with Otsu thresholding and convex hull post processing ...')
    otsu_hull_result_path = os.path.join(analysis_path, 'otsu_hull')
    os.makedirs(otsu_hull_result_path, exist_ok=True)
    list_iou_score, list_dice_score = evaluate_segmentation(parsed_data=parsed_data, prediction=list_otsu_hull_result,
                                           save_path=otsu_hull_result_path)
    logger.info('--Mean IoU score: {}, Dice score: {}\n'.format(str(np.mean(list_iou_score)),
                str(np.mean(list_dice_score))))
