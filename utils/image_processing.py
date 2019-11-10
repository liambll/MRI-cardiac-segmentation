# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:13:02 2019

@author: liam.bui

This file contains functions to create, read, process, and save images
"""

import os
import numpy as np
import cv2
from skimage import transform
        

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


def normalize_image(img):
    """Normalize image to 0-255 range
    :param img: imgge data
    :return normalized_image: image data normalize images to 0-255 range
    """    
    return (img.astype('float') / np.max([np.max(img), 1]) * 255).astype('uint8')

    
def save_images(img_list, save_path, normalize=True):
    """Hortizontally combine a list of images and save it
    :param img_list:list of images, all images should have the same dimension
    :param save_path: str, path to save the combined image
    :param normalize: boolean, whether normalize images to 0-255 range
    """

    if normalize:
        # normalize image to 0-255 scale
        img_list = [normalize_image(img) for img in img_list]
        # put images side by side
        result = np.concatenate(img_list, axis=1).astype('uint8')
    else:
        result = np.concatenate(img_list, axis=1).astype('uint16')

    # save result
    cv2.imwrite(save_path, result)
    
    
def overlay_images(background, masks, colors):
    """Draw boundary of masks on img
    :param background: original background image, grayscale or BGR image
    :param masks: list of binary images, each mask image should have the same dimension as the original image
    :param colors: list of BGR colors to draw for each mask
    :return result: BGR image with 
    """
    
    if len(masks) != len(colors):
        raise ValueError('masks and colors must have the same length.')
    
    # normalize image to 0-255 scale
    background = normalize_image(background)
    masks = [normalize_image(img) for img in masks]
    
    # convert gray scale image to RGB image if needed
    if len(background.shape) < 3:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        
    # create an image to draw boundary on
    result = background.copy()
    for i in range(len(masks)):
        # find boundaries
        _, img_thresh = cv2.threshold(masks[i], 128, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for j in range(len(contours)):
            # creating convex hull object for each contour
            cv2.drawContours(result, contours, j, colors[i], 1)
    
    return result


def save_overlay_images(parsed_data, overlay_path):
    """Draw boundary of masks for all images
    :param parsed_data: list of tuple (img_filename, img_data, imask, omask)
    :param overlay_path: str, path to store images with mask boundaries overlayed
    """    
    
    for (img_filename, img_data, imask, omask) in parsed_data:
        # Overlay with red for i-contour and blue for o-contour
        overlay = overlay_images(img_data, [imask, omask], colors=[(0, 0, 255), (255, 0, 0)])
        save_path = os.path.join(overlay_path, img_filename)
        save_images([overlay], save_path, normalize=True)
        
        
def save_segmentation(parsed_data, prediction, save_path=None):
    """Perform evaluation of segmentation result
    :param parsed_data: list of tuple (img_filename, img_data, imask, omask)
    :param prediction: list, contains predicted i-contour mask
    :param save_path: str, folder path to save segmentation result
    """

    for i in range(len(parsed_data)):
        (img_filename, img_data, imask, omask) = parsed_data[i]
        img_pred = prediction[i]
        overlay = overlay_images(img_data, [imask, img_pred], colors=[(0, 0, 255), (0, 255, 255)])
        save_images([overlay], os.path.join(save_path, img_filename), normalize=True)
    
    
def convex_image(img, largest_hull=False):
    """Get convex hull of a mask
    :param img: binary mask image
    :param largest_hull: bool, whether to get only convex hull with the largest area
    :return result: convex hull of the mask image
    """

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    list_hull = []
    for i in range(len(contours)):
        # creating convex hull object for each contour
        list_hull.append(cv2.convexHull(contours[i], False))
    
    # Draw the convex hull with the largest area
    result = np.zeros(img.shape, np.uint8)
    if largest_hull:
        # Find the convex hull with the largest area
        max_convex_index = np.argmax([cv2.contourArea(i) for i in list_hull])
        cv2.drawContours(result, list_hull, max_convex_index, 255, -1) # -1 indicates fill inside convex hull
    else:
        for i in range(len(list_hull)):
            cv2.drawContours(result, list_hull, i, 255, -1) # -1 indicates fill inside convex hull
        
    return result


def augment_image_pair(img_data, mask):
    """Perform image augmentation with random rotation and flipping
    :param img_data: binary mask image
    :param mask: bool, whether to get only convex hull with the largest area
    :return (img_data, mask): augmented img_data and mask
    """
    # flip vertically
    if np.random.rand() > 0.5:
        img_data = img_data[::-1, :]
        mask = mask[::-1, :]
        
    # flip horizontally
    if np.random.rand() > 0.5:
        img_data = img_data[:, ::-1]
        mask = mask[:, ::-1]
        
    # random rotation
    random_degree = np.random.uniform(-25, 25)
    img_data = transform.rotate(img_data, random_degree)
    mask = transform.rotate(mask, random_degree)
    
    return (img_data, mask)

