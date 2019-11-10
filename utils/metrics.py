# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:04:17 2019

@author: liam.bui

This file contains evaluation metrics

"""

import numpy as np
import keras.backend as K

def iou_score(a, b, smooth=1e-7):
    """Calculate Intersection over Union score:
        IOU = TP / (TP + FN + FN)
    :param a: list of true label
    :param b: list of predicted label
    :param smooth: small value to avoid NaN result
    :return iou: Intersection over Union score
    """  
    a = a.flatten()
    b = b.flatten()

    intersec = a * b
    iou = (np.sum(intersec) + smooth)/ (np.sum(a + b - intersec) + smooth)
    return iou


def dice_score(a, b, smooth=1e-7):
    """Calculate Dice score (also known as F1 score):
        Dice = 2*TP / ((TP + FP) + (TP + FN))
    :param a: list of true label
    :param b: list of predicted label
    :param smooth: small value to avoid NaN result
    :return dice: Dice score
    """
    a = a.flatten()
    b = b.flatten()

    intersec = a * b
    dice = (2*np.sum(intersec) + smooth)/ (np.sum(a + b) + smooth)

    return dice


def k_iou_score(y_true, y_pred, threshold=0.5):
    """Calculate Intersection over Union score as Keras metrics
    :param y_true: list of true label
    :param y_pred: list of predicted probability
    :param threshold: float, threshold to dicide predicted label from probability
    :return Keras IoU score
    """  
    true = K.batch_flatten(y_true)
    pred = K.batch_flatten(y_pred) 
    pred = K.cast(K.greater_equal(pred, threshold), K.floatx())

    intersec = true * pred
    iou = (K.sum(intersec) + K.epsilon())/ (K.sum(true + pred - intersec) + K.epsilon())

    return K.mean(iou)


def k_dice_score(y_true, y_pred, threshold=0.5):
    """Calculate Dice score as Keras metrics
    :param y_true: list of true label
    :param y_pred: list of predicted probability
    :param threshold: float, threshold to dicide predicted label from probability
    :return Keras Dice score
    """  
    true = K.batch_flatten(y_true)
    pred = K.batch_flatten(y_pred) 
    pred = K.cast(K.greater_equal(pred, threshold), K.floatx())

    intersec = true * pred
    dice = (2*K.sum(intersec) + K.epsilon())/ (K.sum(true + pred) + K.epsilon())

    return K.mean(dice)


#def dice_loss(y_true, y_pred):
#    smooth = 1.
#    y_true_f = K.batch_flatten(y_true)
#    y_pred_f = K.batch_flatten(y_pred)
#    intersection = y_true_f * y_pred_f
#    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#    return K.mean(1. - score)


def dice_loss(y_true, y_pred, smooth=1.):
    """Calculate Dice loss as Keras loss
    :param y_true: list of true label
    :param y_pred: list of predicted probability
    :param smooth: small value to avoid NaN result
    :return Keras Dice loss
    """
    
    true = K.flatten(y_true)
    pred = K.flatten(y_pred)
    intersec = true * pred
    score = (2. * K.sum(intersec) + smooth) / (K.sum(true) + K.sum(pred) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    """Calculate Binary Crossentropy + Dice loss as Keras loss
    :param y_true: list of true label
    :param y_pred: list of predicted probability
    :return Keras Binary Crossentropy + Dice loss
    """
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    

def evaluate_segmentation(parsed_data, prediction):
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
        img_pred = prediction[i]/np.max([np.max(prediction[i]), 1]) # Normalize prediction 0-1
        list_iou_score.append(iou_score(imask, img_pred))
        list_dice_score.append(dice_score(imask, img_pred))
            
    return list_iou_score, list_dice_score