# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:04:17 2019

@author: liam.bui

This file contains evaluation metrics

"""

import numpy as np

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