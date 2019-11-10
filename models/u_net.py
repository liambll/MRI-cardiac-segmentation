# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:24:32 2019

@author: liam.bui

This file contains code for u-net model
"""

import numpy as np
import logging
from utils import dataset

from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model


class UNet:
    def __init__(self, img_dim=256, img_channel=1, logger=None):
        """
        :param img_dim: int, image dimension, must be divisible by 16
        :param img_channel: int, number of image channels: 1 for grayscale, 3 for RGB images
        :param logger: Logger object
        """
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('u-net')
        else:
            self.logger = logger
        self.model = None
        self.loss_function = None
        self.metric_list = []
        self.creating_model(img_dim=img_dim, img_channel=img_channel)

    def creating_model(self, img_dim=256, img_channel=1):
        """ Create model architecture
        :param img_dim: int, image dimension, must be divisible by 16
        :param img_channel: int, number of image channels: 1 for grayscale, 3 for RGB images

        Code is adopted from https://github.com/zhixuhao/unet/blob/master/model.py
        VGG16 layer name is added so that we can load pre-trained ImageNet weights
        """

        self.logger.info('Building model ...')
        if img_channel not in [1, 3]:
            raise ValueError('Image channel must be 1 (grayscale) or 3 (RGB).')
        if img_dim % 16 != 0:
            raise ValueError('Image dimension must be divisable by 16.')

        # Prepare image for input to model
        img_input = Input([img_dim, img_dim, img_channel])

        # # Block 1
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)  # dimension = img_dim/2

        # Block 2
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)  # dimension = img_dim/4

        # Block 3
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)  # dimension = img_dim/8

        # Block 4
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)  # dimension = img_dim/16

        # Block 5
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5)

        # Upsampling block
        up6 = Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv5))  # dimension = img_dim/8
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))  # dimension = img_dim/4
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))  # dimension = img_dim/2
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))  # dimension = img_dim
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        self.model = Model(input=img_input, output=conv10)
        self.logger.info('Finish building model.')

    def load_weights(self, weights_path):
        """ Create model architecture
        :param weights_path: str, filepath to h5 model weight
        """

        self.logger.info('Loading model weight ...')
        self.model.load_weights(weights_path, by_name=True)
        self.logger.info('Finish loading model weight.')

    def compile_model(self, optimizer, loss_function='binary_crossentropy', metric_list=['accuracy']):
        """ Create model architecture
        :param optimizer: str to indicate a supported Keras optimizer, or a custom Keras optimizer object
        :param loss_function: str to indicate a supported Keras loss function, or a custom Keras loss function
        :param metric_list: list of metrics,
            each metric is a str to indicate a supported Keras metrics, or a custom Keras metrics function
        """

        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metric_list)
        self.loss_function = loss_function
        self.metric_list = metric_list

    def fit(self, train_data, val_data, nb_epochs, batch_size, save_path):
        """Fit model
        :param train_data: list of tuples (img_id, img_data, imask, omask)
        :param val_data: list of tuples (img_id, img_data, imask, omask)
        :param nb_epochs: int, number of training epoch
        :param batch_size: int, batch size for training
        :param save_path: str, path to save model weight
        """

        if self.loss_function is None:
            raise ValueError('Model is not compiled.')

        best_val_loss = float('inf')
        nb_epochs_no_improvement = 0
        metric_names = [x if type(x) == str else x.__name__ for x in [self.loss_function] + self.metric_list]
        nb_train_steps = int(np.ceil(len(train_data) / batch_size))

        self.logger.info('Start training ...')
        for epoch in range(nb_epochs):
            self.logger.info('Epoch {}/{}:'.format(str(epoch), str(nb_epochs)))
            # Train step
            train_metric_values = [0] * len(metric_names)
            nb_train_samples = 0
            train_gen = dataset.data_generator(datasource=train_data, batch_size=batch_size, shuffle=True,
                                               augmentation=True, infinite_loop=False, logger=None)
            for step in range(nb_train_steps):
                batch_img, batch_mask = train_gen.__next__()
                nb_train_samples += len(batch_img)
                batch_metrics = self.model.train_on_batch(batch_img, batch_mask)
                train_metric_values = [train_metric_values[i] + batch_metrics[i] * len(batch_img)
                                       for i in range(len(metric_names))]
            train_metric_values = [i / nb_train_samples for i in train_metric_values]
            train_metric_txt = ', '.join([str(a) + ':' + str(b) for (a, b) in zip(metric_names, train_metric_values)])
            self.logger.info('---Train {}'.format(train_metric_txt))

            # Val step
            val_metric_values = self.evaluate(data_source=val_data, batch_size=batch_size)
            val_metric_txt = ', '.join([str(a) + ':' + str(b) for (a, b) in zip(metric_names, val_metric_values)])
            self.logger.info('---Val {}.\n'.format(val_metric_txt))

            # Save best model weights
            if val_metric_values[0] < best_val_loss:  # val_metric_values[0] is val loss
                best_val_loss = val_metric_values[0]
                nb_epochs_no_improvement = 0
                self.model.save_weights(save_path)
            else:
                nb_epochs_no_improvement += 1

            if nb_epochs_no_improvement > 30:  # Early stop if val loss does not improve after 30 epochs
                self.logger.info('Early stop at epoch {}.\n'.format(epoch))
                break

    def evaluate(self, data_source, batch_size):
        """Fit model
        :param data_source: list of tuples (img_id, img_data, imask, omask)
        :param batch_size: int, batch size for evaluation
        """

        data_gen = dataset.data_generator(datasource=data_source, batch_size=batch_size, shuffle=False,
                                          augmentation=False, infinite_loop=False, logger=None)
        nb_steps = int(np.ceil(len(data_source) / batch_size))

        metric_names = [x if type(x) == str else x.__name__ for x in [self.loss_function] + self.metric_list]
        val_metric_values = [0] * len(metric_names)
        nb_val_samples = 0
        for step in range(nb_steps):
            batch_img, batch_mask = data_gen.__next__()
            nb_val_samples += len(batch_img)
            batch_metrics = self.model.test_on_batch(batch_img, batch_mask)
            val_metric_values = [val_metric_values[i] + batch_metrics[i] * len(batch_img)
                                 for i in range(len(metric_names))]
        val_metric_values = [i / nb_val_samples for i in val_metric_values]

        return val_metric_values

    def predict(self, data_source, batch_size):
        """Fit model
        :param data_source: list of tuples (img_id, img_data, imask, omask)
        :param batch_size: int, batch size for evaluation
        :return prediction: ndarray, predicted mask
        """

        data_gen = dataset.data_generator(datasource=data_source, batch_size=batch_size, shuffle=False,
                                          augmentation=False, infinite_loop=False, logger=None)
        nb_steps = int(np.ceil(len(data_source) / batch_size))
        prediction = []
        for step in range(nb_steps):
            batch_img, batch_mask = data_gen.__next__()
            batch_predict = self.model.predict_on_batch(batch_img)
            prediction.append(batch_predict)

        return np.concatenate(prediction, axis=0)
