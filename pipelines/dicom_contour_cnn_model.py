# -*- coding: utf-8 -*-
"""
Pipeline to train and evaluate convolutional neural network models
"""

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
set_session(tf.Session(config=config))
from keras.optimizers import Adam

from utils import metrics, image_processing
from models import u_net
import argparse
import logging
import numpy as np


if __name__ == '__main__':
    # parse parameters
    parser = argparse.ArgumentParser(description='Analysis of dicom and contour files')
    parser.add_argument('data_path', help='A folderpath for parsed dicom-contour images')
    parser.add_argument('model_path', help='A folderpath to store model weight and result')
    parser.add_argument('--train', action='store_true', help='Whether to train model')
    parser.add_argument('--nb_epochs', type=int, default=100, help='Number of epochs to train model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to train model')
    args = parser.parse_args()
    
    data_path = args.data_path
    model_path = args.model_path
    train = args.train
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    
    # Initiate logger
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler('dicom_contour_cnn_model.log'),
                                                      logging.StreamHandler()])
    
    logger = logging.getLogger('dicom_contour_cnn_model')

    # Read parsed data
    logger.info('Reading parsed data ...')
    parsed_data = image_processing.read_parsed_images(data_path, img_dim=256)
    
    # Perform train val split by patient_ids
    logger.info('Performing train/val split ...')
    unique_patient_ids = np.unique([img_id.split('_')[0] for (img_id, _, _, _) in parsed_data])
    train_data = [obervation for obervation in parsed_data if obervation[0].split('_')[0] in unique_patient_ids[:3]]
    val_data = [obervation for obervation in parsed_data if obervation[0].split('_')[0] in unique_patient_ids[3:]]

    # Create checkpoint for model weight
    checkpoint_path = os.path.join(model_path, 'checkpoint')
    save_path = os.path.join(checkpoint_path, 'best_model.h5')    
    os.makedirs(checkpoint_path, exist_ok=True)
    
    unet = u_net.UNet(img_dim=256, img_channel=3, logger=logger)
    if train: # Traning is required      
        # Set up model
        vgg_weights = os.path.join(checkpoint_path, 'vgg_weights.h5')
        if os.path.exists(vgg_weights):
            unet.load_weights(vgg_weights) # Load ImageNet weights
        unet.compile_model(optimizer=Adam(lr=1e-4), loss_function=metrics.bce_dice_loss,
                     metric_list=['accuracy', metrics.k_dice_score, metrics.k_iou_score])
        
        # Train model
        unet.fit(train_data=train_data, val_data=val_data,
                           nb_epochs=nb_epochs,
                           batch_size=batch_size,
                           save_path=save_path)
  
    # Evaluate model
    logger.info('Evaluating U-Net model ...')
    unet.load_weights(save_path)
    predictions = unet.predict(parsed_data, batch_size=batch_size)
    predicted_labels = np.squeeze((predictions > 0.5).astype('uint8'))
    
    cnn_result_path = os.path.join(model_path, 'u_net')
    os.makedirs(cnn_result_path, exist_ok=True)
    list_iou_score, list_dice_score = metrics.evaluate_segmentation(parsed_data=parsed_data,
                                                                    prediction=predicted_labels)
    image_processing.save_segmentation(parsed_data=parsed_data, prediction=predicted_labels,
                                       save_path=cnn_result_path)
    logger.info('--IoU Score: Mean {}, Std {}\n--Dice score: Mean {}, Std {}\n'.format(str(np.mean(list_iou_score)),
                str(np.std(list_iou_score)), str(np.mean(list_dice_score)), str(np.std(list_dice_score))))

