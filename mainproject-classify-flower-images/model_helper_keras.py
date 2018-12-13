#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Vittorio Nardone
# DATE CREATED: 12/13/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Train a new deep neural network on a image dataset and save
#          the model as a checkpoint.
#          Use a trained network to predict the class for an input image
#          loading checkpoint model
#          Pre-trained networks are used (VGG/Densenet/AlexNet/ResNet)
#

#import torch
#from torch import nn
#from torch import optim
#import torch.nn.functional as F
#from torch.autograd import Variable
#from torchvision import datasets, transforms, models

import keras
import keras.applications as app
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import keras.optimizers as optim
from keras.preprocessing.image import ImageDataGenerator

from collections import OrderedDict
import numpy as np
import os.path
from time import time, gmtime, strftime
#from PIL import Image

import helper

def supported_models():
    ''' Return a list of supported models architecture
    '''
    return ['MobileNet']

def supported_optimizer():
    ''' Return a list of supported optimizer
    '''
    return ['SGD','Adam']

def gpu_available():
    ''' Return True if Cuda/GPU is available on current system
    '''
    #TODO
    return False

def create_new_model(arch):
    """
    Create a new pretrained model with specified architecture.
    Parameters:
     arch - model architecture
    Returns:
     model - the model object
    """
    # Model definition
    if arch in supported_models():
        if arch == 'MobileNet':
            model = app.mobilenet.MobileNet(weights='imagenet', include_top=False)
    else:
        raise Exception("Unknow architecture: {}".format(arch))

    return model


def create_classifier(base_model, hidden_units, class_count):
    """
    Create classifier according to specified parameters and change it inplace.
    Parameters:
     base_model - the model object
     hiddend_units - int array with number of elements for each hidden layer
     class_count - number of output layer elements
    Returns:
     model
    """
    input_count = int(base_model.output.shape[-1])

    # Input and output layers
    hidden_units.insert(0, input_count)
    hidden_units.append(class_count)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    iterations = len(hidden_units)

    for idx in range(iterations):
        if idx < iterations-1:
            x = Dense(hidden_units[idx], activation='relu')(x)
        else:
            x = Dense(hidden_units[idx], activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)

    return model


def create_optimizer(optimization = 'SGD', learning_rate = 0.05):
    if optimization == 'SGD':
        optimizer = optim.SGD(lr=learning_rate)
    elif optimization == 'Adam':
        optimizer = optim.Adam(lr=learning_rate)
    else:
        raise Exception("Unknow optimization algorithm")
    return optimizer

def create_and_train(data_folder,
                     training_subfolder = '/train/',
                     validation_subfolder = '/valid/',
                     batch_size = 64,
                     arch = 'densenet121', hidden_units = [], dropout = 0, bn = False,
                     epochs =  10, learning_rate = 0.05, accuracy = 0.8, optimization = 'SGD',
                     gpu_mode = False, full_net_epochs = 0):
    """
    Create a model and train it according to specified parameters.
    Parameters:
     data_folder - root image directory
     training_subfolder - training images subfolder
     validation_subfolder - validation images subfolder
     arch - model architecture
     hiddend_units - int array with number of elements for each classifier hidden layer
     dropout - dropout probability in classifier
     epochs - number of epochs to run
     learning_rate - learning rate to be used in optimizer
     accuracy - validation accuracy threshold
     gpu_mode - use GPU if True
     optimization - set optimization algorithm (Adam/SGD)
     bn - add Batch Normalization layers in classifier
     full_net_epochs - Retrain all elements of network after classifier training for specified epochs
    Returns:
     model - the trained model object
    """
    # Set data folder to current directory if empty
    if (data_folder == ''):
        data_folder = '.'

    # Dataset loading
    train_generator = ImageDataGenerator(
              preprocessing_function=keras.applications.mobilenet.preprocess_input
              ).flow_from_directory(os.path.normpath(data_folder + training_subfolder), target_size=(224, 224),
          class_mode='categorical', batch_size=batch_size)

    valid_generator = ImageDataGenerator(
              preprocessing_function=keras.applications.mobilenet.preprocess_input
              ).flow_from_directory(os.path.normpath(data_folder + validation_subfolder), target_size=(224, 224),
          class_mode='categorical', batch_size=batch_size)


    # Create model
    print("Model architecture: '{}'".format(arch), "- GPU Mode:", gpu_mode)
    base_model = create_new_model(arch)

    # Create a new Classifier
    model = create_classifier(base_model, hidden_units, len(set(train_generator.classes)))

    for layer in base_model.layers:
        layer.trainable = False

    # Optimizer definition
    optimizer = create_optimizer(optimization = optimization,
                                 learning_rate = learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy')


    print("Learning rate:", learning_rate, "- Criterion: CrossEntropyLoss", "- Optimizer:", optimization)
    print("Batch Size: {} - Training stop after {} epoch(s)".format(batch_size, epochs), end = ' ')
    if accuracy < 1:
        print("or validation accuracy > {}%".format(accuracy*100), end = ' ')
    print()

    
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.n//train_generator.batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.n//valid_generator.batch_size,
                        epochs=epochs)

    # Save configuration for futher use
    model.config = { 'arch' : arch,
                     'hidden_units' : hidden_units,
                     'dropout' : dropout,
                     'class_count' : len(set(train_generator.classes)),
                     'class_to_idx': training_data.class_to_idx,
                     'gpu_mode' : gpu_mode,
                     'train_epoch' : epochs,
                     'learning_rate' : learning_rate,
                     'bn' : bn,
                   }

    return model

def save_checkpoint(model, destination_folder, filename = ""):
    """
    Save model checkpoint to file.
    Parameters:
     model - the model object
     destination_folder - destination folder of checkpoint file
     filename - checkpoint filename
    Returns:
     filename - checkpoint filename
    """
    #TODO
    #checkpoint = {'model_state_dict': model.state_dict(),
    #              'model_config': model.config
    #             }

    # Compose filename
    if filename == "":
        filename = "cp_{}_e{}_lr{}.pth".format(model.config['arch'],
                                               model.config['train_epoch'],
                                               model.config['learning_rate'])
    if destination_folder == "":
        destination_folder = '.'

    full_filename = os.path.normpath("{}/{}".format(destination_folder,filename))
    #TODO
    #torch.save(checkpoint, full_filename)
    print("\n** Model checkpoint saved to: '{}'".format(full_filename))
    return filename

def create_model_from_checkpoint(filename):
    """
    Create new model from checkpoint file.
    Parameters:
     filename - checkpoint filename
    Returns:
     model - the model object
    """
    pass

# Scales, crops, and normalizes a PIL image for a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Resize & Crop
    image.thumbnail((256,256))
    left, top, right, bottom = (image.size[0] - 224)/2, (image.size[1] - 224)/2, (image.size[0] + 224)/2, (image.size[1] + 224)/2
    img_np = np.array(image.crop((left, top, right, bottom)))

    #Normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np/255.0 - mean) / std

    #Channels order
    img_np = img_np.transpose((2,0,1))

    return img_np

# Predict the class (or classes) of an image
def predict(image_path, model, topk=5, gpu_mode = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''

    pass


# TODO: add module sanity check
if __name__ == "__main__":
    pass
