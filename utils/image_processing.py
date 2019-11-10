# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:13:02 2019

@author: liam.bui

This file contains functions to create, process, and save images
"""

import numpy as np
import cv2
        

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
        result = np.concatenate(img_list, axis=1)

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