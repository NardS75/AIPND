#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Vittorio Nardone
# DATE CREATED: 05/07/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Use a trained network to predict the class for an input image
#          loading checkpoint model
#          Pre-trained networks are used (VGG/Densenet/AlexNet)
#
# Expected Call with <> indicating expected user input:
#      python predict.py <input> <checkpoint>
#             --top_k <top_k>
#             --category_names <filename>
#             --gpu
#
# Example call:
#    python predict.py testimg.jpg densenet121.pth --category_names cat_to_name.json --gpu
#    python predict.py testimg.jpg checkpoints/cp_vgg11_e5_lr0.05.pth --top_k 5 --category_names cat_to_name.json
#
# Arguments explaination:
# <input> (required)
#     Input image filename
#
# <checkpoint> (required)
#     Model checkpoint filename, created using train.py command line application
#
# --top_k <top_k> (optional, default is 1)
#     Return top_K most likely classes
#
# --category_names <filename> (optional, default is no file)
#     Use a mapping of categories to real names. File must be in json format
#
# --gpu (optional, default "cpu")
#     If set, GPU is used to predict the class of input image
#
#
# Imports python modules
import argparse
import os.path
from time import time, gmtime, strftime

import model_helper_keras as mhk
import model_helper_pytorch as mhp
import helper


# Main program function defined below
def main():
    #Collect start time
    start_time = time()

    #Parse command line arguments
    in_arg = get_input_args()
    
    #Load categories
    if in_arg.category_names != '':
        cat_to_name = helper.load_category_names(in_arg.category_names)    

    #Load model checkpoint
    if in_arg.backend == 'keras':
        model = mhk.create_model_from_checkpoint(in_arg.checkpoint)
    elif in_arg.backend == 'pytorch':
        model = mhp.create_model_from_checkpoint(in_arg.checkpoint)
    else:
        raise Exception("Unknow backend: {}".format(in_arg.backend))

    # Get file list
    if os.path.isfile(in_arg.input):
        file_list = [in_arg.input]
    else:
        file_list = [os.path.join(in_arg.input, f) for f in os.listdir(in_arg.input) if os.path.isfile(os.path.join(in_arg.input, f))]  
       
    # Make predictions
    for filename in file_list:    

        #Prediction
        if in_arg.backend == 'keras':
            probs, classes = mhk.predict(filename, model, in_arg.top_k)        
        elif in_arg.backend == 'pytorch':
            probs, classes = mhp.predict(filename, model, in_arg.top_k, gpu_mode = in_arg.gpu)        
        
        print("\n** Top {} prediction result for filename '{}'".format(in_arg.top_k, filename))

        result = 0
        for c,p in zip(classes, probs):
            result += 1
            print("{0:2}) Class: {1:3} - Prob: {2:5.2f}%".format(result, c, p*100), end = ' ')
            if in_arg.category_names != '':
                if c in cat_to_name:
                    print("- Category: '{}'".format(cat_to_name[c].title()), end = ' ')
                else:
                    print("- Category: (not found)", end = ' ')
            print()

    tot_time = time() - start_time
    print("\n** Total Elapsed Runtime:", strftime("%H:%M:%S", gmtime(tot_time)))


# Command line arguments parser
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # create arguments parser
    parser = argparse.ArgumentParser()

    parser.add_argument("input", nargs=1, type = str,
                    help = "Input image filename or images directory")
    parser.add_argument("checkpoint", nargs=1, type = str,
                    help = "Model checkpoint filename, created using train.py command line application")
    parser.add_argument('--top_k', type = int, default = 1,
                    help = "Return top_K most likely classes (default: 1)")
    parser.add_argument('--category_names', type = str, default = '',
                    help = "Use a mapping of categories to real names. File must be in json format (default: no file)")
    parser.add_argument('--gpu', action='store_true',
                    help = "If set, GPU is used in prediction (default: False)")

    in_arg = parser.parse_args()

    in_arg.input = in_arg.input[0]
    in_arg.checkpoint = in_arg.checkpoint[0]
    in_arg.backend = helper.get_backend(in_arg.checkpoint)

    error_list = []

    # Check top_k
    if in_arg.top_k <= 0:
        error_list.append("predict.py: error: argument: --top_k: must be positive int")

    # Check input Filename
    if not os.path.exists(in_arg.input):
        error_list.append("predict.py: error: argument: input: file/directory not found '{}'".format(in_arg.input))

    # Check checkpoint Filename
    if not os.path.isfile(in_arg.checkpoint):
        error_list.append("predict.py: error: argument: checkpoint: file not found '{}'".format(in_arg.checkpoint))

    # Check categories Filename
    if in_arg.category_names != '':
        if not os.path.isfile(in_arg.category_names):
            error_list.append("predict.py: error: argument: --category_names: file not found '{}'".format(in_arg.category_names))

    # Check GPU
    if in_arg.backend == 'pytorch':
        if in_arg.gpu and not mhp.gpu_available():
            error_list.append("predict.py: error: argument: --gpu: GPU not available")

    # Print errors
    if len(error_list) > 0:
        parser.print_usage()
        print('\n'.join(error for error in error_list))
        quit()
        
    # return arguments object
    return in_arg

# Call to main function to run the program
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n** User interruption")
        quit()
