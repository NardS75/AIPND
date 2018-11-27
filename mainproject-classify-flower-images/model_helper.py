#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Vittorio Nardone
# DATE CREATED: 05/07/2018
# REVISED DATE: 11/21/2018            <=(Date Revised - if any)
# PURPOSE: Train a new deep neural network on a image dataset and save
#          the model as a checkpoint.
#          Use a trained network to predict the class for an input image
#          loading checkpoint model
#          Pre-trained networks are used (VGG/Densenet/AlexNet/ResNet)
#

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import os.path
from time import time, gmtime, strftime
from PIL import Image

import helper

def supported_models():
    ''' Return a list of supported models architecture
    '''
    return ['densenet121', 'densenet161', 'densenet169', 'densenet201',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
            'alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def supported_optimizer():
    ''' Return a list of supported optimizer
    '''
    return ['SGD','Adam']

def gpu_available():
    ''' Return True if Cuda/GPU is available on current system
    '''
    return torch.cuda.is_available()

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
        model = getattr(models, arch)(pretrained=True)
    else:
        raise Exception("Unknow architecture: {}".format(arch))

    for param in model.parameters():
        param.requires_grad = False

    return model


def create_classifier(model, hidden_units, class_count, dropout = 0, skip_add = False, bn = False, bn_momentum = 0.01, print_desc = True):
    """
    Create classifier according to specified parameters and change it inplace.
    Parameters:
     model - the model object
     hiddend_units - int array with number of elements for each hidden layer
     class_count - number of output layer elements
     dropout - dropout probability, if 0 dropout is not added in classifier
     skip_add - if True, does not add input/output layers (useful when loading checkpoint)
     bn - if True, add a BatchNorm1d layer
     bn_momentum - BatchNorm1d momentum setting
    Returns:
     None
    """
    if not skip_add:
        # Find model original classifier name
        if hasattr(model, 'classifier'):
            orig_classifier = model.classifier
        elif hasattr(model, 'fc'):
            orig_classifier = model.fc
        else:
            raise Exception("Unknow classifier name")

        # Check model original classifier input count
        if type(orig_classifier) == torch.nn.modules.container.Sequential:
            #Get first Linear
            for idx in np.arange(len(orig_classifier)):
                if type(orig_classifier[idx]) == torch.nn.modules.linear.Linear:
                    input_count = orig_classifier[idx].in_features
                    break
        elif type(orig_classifier) == torch.nn.modules.linear.Linear:
            input_count = orig_classifier.in_features
        else:
            raise Exception("Unknow classifier type: {}".format(type(orig_classifier)))
        # Input and output layers
        hidden_units.insert(0, input_count)
        hidden_units.append(class_count)

    classifier_list = []
    iterations = len(hidden_units)-1

    for idx in np.arange(iterations):
        classifier_list.append(('fc{}'.format(idx+1), nn.Linear(hidden_units[idx], hidden_units[idx+1])))
        # Add ReLU
        if idx < iterations-1:
            classifier_list.append(('relu{}'.format(idx+1), nn.ReLU()))
            # Add Dropout
            if (dropout > 0):
                classifier_list.append(('drop{}'.format(idx+1), nn.Dropout(p=dropout)))
            # Add Batch Normalization
            if bn:
                classifier_list.append(('bn{}'.format(idx+1), nn.BatchNorm1d(hidden_units[idx+1], momentum=bn_momentum)))

    classifier = nn.Sequential(OrderedDict(classifier_list))

    if print_desc:
        print("Classifier:", classifier)

    # Find model original classifier and replace it
    if hasattr(model, 'classifier'):
        model.classifier = classifier
    elif hasattr(model, 'fc'):
        model.fc = classifier

def model_train_epoch(model, criterion, optimizer, training_loader, gpu_mode):
    """
    Train a model (1 epoch) according to specified parameters.
    Parameters:
     model - model object
     criterion - loss function
     optimizer - optimizer function
     training_loader - train dataset loader
     gpu_mode - use GPU if True
    Returns:
     model - the trained model object
     loss - epoch loss
    """
    #Trainining
    model.train()
    running_loss = 0
    steps, steps_total = 0, len(training_loader)

    for inputs, labels in iter(training_loader):
        steps += 1
        helper.printProgressBar(steps, steps_total, prefix = 'Progress:', decimals = 0, length = 60)

        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        if gpu_mode:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return model, running_loss / steps

def model_validation(model, criterion, validation_loader, gpu_mode):
    """
    Validate a model according to specified parameters.
    Parameters:
     model - model object
     criterion - loss function
     validation_loader - validation dataset loader
     gpu_mode - use GPU if True
    Returns:
     loss - validation loss
     accuracy - validation accuracy
    """
    #Validation
    model.eval()

    running_loss = 0
    steps, steps_total = 0, len(validation_loader)

    correct_preds = 0
    tot_preds = 0

    for inputs, labels in iter(validation_loader):
        steps += 1
        helper.printProgressBar(steps, steps_total, prefix = 'Progress:', decimals = 0, length = 60)

        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        if gpu_mode:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Forward pass only
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        ps = F.softmax(outputs, dim=1)
        if gpu_mode:
            ps = ps.cpu()
        pred = np.argmax(ps.data.numpy(), axis=1)
        correct_labels = labels.cpu().data.numpy()
        correct_preds += np.sum(correct_labels == pred)
        tot_preds += pred.shape[0]

    return running_loss / steps, correct_preds / tot_preds

def create_optimizer(model, optimization = 'SGD', learning_rate = 0.05, full_net = False):
    if optimization == 'SGD':
        if full_net:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            if hasattr(model, 'classifier'):
                optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
            elif hasattr(model, 'fc'):
                optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate)
            else:
                raise Exception("Unknow classifier name")
    elif optimization == 'Adam':
        if full_net:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            if hasattr(model, 'classifier'):
                optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            elif hasattr(model, 'fc'):
                optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
            else:
                raise Exception("Unknow classifier name")
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
    # Transforms definition - Training (random rotation / crop / flip)
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    # Transforms definition - Validation (resize / crop)
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])

    # Set data folder to current directory if empty
    if (data_folder == ''):
        data_folder = '.'

    # Dataset loading
    training_data = datasets.ImageFolder(os.path.normpath(data_folder + training_subfolder), transform=training_transforms)
    validation_data = datasets.ImageFolder(os.path.normpath(data_folder + validation_subfolder), transform=validation_transforms)

    # Batch loader definition
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)

    # Create model
    print("Model architecture: '{}'".format(arch), "- GPU Mode:", gpu_mode)
    model = create_new_model(arch)

    # Create a new Classifier
    create_classifier(model, hidden_units, len(training_data.classes), dropout = dropout, bn = bn)

    # Criterion and optimizer definition
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimization = optimization,
                                 learning_rate = learning_rate)

    print("Learning rate:", learning_rate, "- Criterion: CrossEntropyLoss", "- Optimizer:", optimization)
    print("Batch Size: {} - Training stop after {} epoch(s)".format(batch_size, epochs), end = ' ')
    if accuracy < 1:
        print("or validation accuracy > {}%".format(accuracy*100), end = ' ')
    print()

    # Move to GPU
    if gpu_mode:
        model.cuda()

    # Training & Validation
    epoch_count = 0
    validation_accuracy = 0
    train_loss = 0
    valid_loss = 0

    while (epoch_count < epochs) and (validation_accuracy < accuracy):
          epoch_count += 1

          epoc_time = time()
          print("\n** Training epoch {}/{} BEGIN **".format(epoch_count, epochs))
          model, train_loss = model_train_epoch(model, criterion, optimizer, training_loader, gpu_mode)
          print("Epoch duration:", strftime("%H:%M:%S", gmtime(time() - epoc_time)), "- Loss on training dataset: {:.4f}".format(train_loss))
          print("** Training epoch {}/{} END **".format(epoch_count, epochs))

          print("\n** Validation after {} epoch(s) **".format(epoch_count))
          valid_loss, validation_accuracy = model_validation(model, criterion, validation_loader, gpu_mode)
          print("Loss on validation dataset: {:.4f}".format(valid_loss), "- Accuracy: {:.2f}%".format(validation_accuracy*100))

    if (epoch_count < epochs):
          print("\n** Validation accuracy threshold reached")
    elif full_net_epochs > 0:
          print("\n** Full network training BEGIN **")
          # Retrain whole network
          for param in model.parameters():
              param.requires_grad = True

          epoch_count = 0
          optimizer = create_optimizer(model, optimization = optimization,
                                       learning_rate = learning_rate,
                                       full_net = True)

          while (epoch_count < full_net_epochs) and (validation_accuracy < accuracy):
              epoch_count += 1

              epoc_time = time()
              print("\n** Training epoch {}/{} BEGIN **".format(epoch_count, epochs))
              model, train_loss = model_train_epoch(model, criterion, optimizer, training_loader, gpu_mode)
              print("Epoch duration:", strftime("%H:%M:%S", gmtime(time() - epoc_time)), "- Loss on training dataset: {:.4f}".format(train_loss))
              print("** Training epoch {}/{} END **".format(epoch_count, epochs))

              print("\n** Validation after {} epoch(s) **".format(epoch_count))
              valid_loss, validation_accuracy = model_validation(model, criterion, validation_loader, gpu_mode)
              print("Loss on validation dataset: {:.4f}".format(valid_loss), "- Accuracy: {:.2f}%".format(validation_accuracy*100))

          print("\n** Full network training END **")


    # Save configuration for futher use
    model.config = { 'arch' : arch,
                     'hidden_units' : hidden_units,
                     'dropout' : dropout,
                     'class_count' : len(training_data.classes),
                     'class_to_idx': training_data.class_to_idx,
                     'gpu_mode' : gpu_mode,
                     'train_epoch' : epoch_count,
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
    checkpoint = {'model_state_dict': model.state_dict(),
                  'model_config': model.config
                 }

    # Compose filename
    if filename == "":
        filename = "cp_{}_e{}_lr{}.pth".format(model.config['arch'],
                                               model.config['train_epoch'],
                                               model.config['learning_rate'])
    if destination_folder == "":
        destination_folder = '.'

    full_filename = os.path.normpath("{}/{}".format(destination_folder,filename))
    torch.save(checkpoint, full_filename)
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
    #Load checkpoint file. "map_location" is used to allow CPU load() of GPU generated checkpoints
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    #Create a new pre-trained model with same architecture
    print("Model architecture: '{}'".format(checkpoint['model_config']['arch']),
          "- checkpoint loaded from: '{}'".format(filename))
    model = create_new_model(checkpoint['model_config']['arch'])

    #Create a new classifier with checkpoint configuration

    #backward compatibility - chech if 'bn' key is present in model configuration
    bn = False
    if 'bn' in checkpoint['model_config']:
        bn = checkpoint['model_config']['bn']

    create_classifier(model,
                      checkpoint['model_config']['hidden_units'],
                      checkpoint['model_config']['class_count'],
                      skip_add = True,
                      print_desc = False,
                      bn = bn)

    #Loading state and classe idxs
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['model_config']['class_to_idx']
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    return model

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

    #Load and pre-process image
    img = Image.open(image_path)
    img_np = process_image(img)
    img_np = img_np[np.newaxis,:].astype('f')

    inputs = Variable(torch.from_numpy(img_np), requires_grad=False)

    #Move to GPU
    if gpu_mode:
        model.cuda()
        inputs = inputs.cuda()
    else:
        model.cpu()

    #Forward pass only
    model.eval()
    outputs = model.forward(inputs)
    ps = F.softmax(outputs, dim=1)

    #Get topk
    probs, classes_idx = ps.topk(topk)

    #Move back to CPU
    if gpu_mode:
        probs, classes_idx = probs.cpu(), classes_idx.cpu()

    #Idx to Class
    classes_idx = classes_idx.data.numpy().squeeze(0)
    classes = [model.idx_to_class[k] for k in classes_idx]

    return probs.data.numpy().squeeze(0), classes


# TODO: add module sanity check
if __name__ == "__main__":
    pass
