#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Vittorio Nardone
# DATE CREATED: 05/07/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Generic helper functions

import json

# Print iterations progress
# Orginal source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


# Load category names from a json file
def load_category_names(filename):
    """Load category names, return dict label -> name"""
    with open(filename, 'r') as f:
        return json.load(f)

# Detect backend by checkpoint file extension
def get_backend(checkpoint_file):
    file_ext = checkpoint_file.rsplit('.', 1)[-1]
    backend  = ''
    
    if file_ext == 'pth':
        backend = 'pytorch'
    elif file_ext == 'h5':
        backend = 'keras'
    else:
        raise Exception("Unknown checkpoint file extension: {}".format(file_ext))
    
    return backend

# Call to main function to run the program
if __name__ == "__main__":
    pass
