"""
    Script to convert afew2018 dataset into a 0-6 facial emotion labels folder 
    convention as part of the data preparation process to convert to numpy 
    format or merge with other datasets for CNN baseline model training and 
    testing.

    ## USAGE ##
    
    1. Within afew2018 dataset folder, extract the following zip files 
    ("Train_AFEW.zip", "Val_AFEW.zip").
    2. Within both "Train_AFEW.zip" and "Val_AFEW.zip" folder, extract 
    the following files ("AlignedFaces_LBPTOP_Points.zip", "AlignedFaces_LBPTOP_
    Points_Val.zip").
    3. Move this script into the root afew2018 folder and run the script.
"""

import os
import cv2
import csv
import zipfile
import rarfile
import shutil
import itertools
import time
import glob
from tqdm import tqdm
from collections import Counter

ROOT_DIR = os.getcwd()

NEUTRAL_FOLDER = ROOT_DIR + '\\labelled_image\\0'
ANGER_FOLDER = ROOT_DIR + '\\labelled_image\\1'
DISGUST_FOLDER = ROOT_DIR + '\\labelled_image\\2'
FEAR_FOLDER = ROOT_DIR + '\\labelled_image\\3'
HAPPY_FOLDER = ROOT_DIR + '\\labelled_image\\4'
SAD_FOLDER = ROOT_DIR + '\\labelled_image\\5'
SURPRISE_FOLDER = ROOT_DIR + '\\labelled_image\\6'

CROPPED_IMG_FOLDER = ROOT_DIR + '\\cropped_images'

LABEL_LIST = [[], [], [], [], [], [], []]
IMAGE_FILEPATH = []
IMAGE_LABEL_LIST = [[], [], [], [], [], [], []]

def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def _list_files(target_dir, owd):
    
    files = []

    os.chdir(target_dir)
    files = os.listdir()
    os.chdir(owd)

    return files

def _rem_extn_from_list(list):

    """
        Removes file extension from items in a list.
    """

    new_list = []

    for i in range(len(list)):
         _labels = os.path.splitext(list[i])[0]
         new_list.append(_labels)

    return new_list

def get_images(dir, img_filepath):

    """
        Go through specified directory to look for all images' filepath.
    """

    os.chdir(dir)

    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            if file.endswith(".jpg"):
                img_filepath.append(filepath)

    os.chdir(ROOT_DIR)

def get_labels(dir, label_list):

    """
        Go through specified directory to look for all labelled folder with
        associated images.

        label_list format:  [[angry..], [Disgust], ...., [Surprise]]
    """

    os.chdir(dir)

    for subdir, dirs, files in os.walk(dir):
        for _dir in dirs:

            if _dir == "Angry":
                _files = _list_files(_dir, dir)
                _labels = _rem_extn_from_list(_files)
                label_list[0].extend(_labels)
            elif _dir == "Disgust":
                _files = _list_files(_dir, dir)
                _labels = _rem_extn_from_list(_files)
                label_list[1].extend(_labels)
            elif _dir == "Fear":
                _files = _list_files(_dir, dir)
                _labels = _rem_extn_from_list(_files)
                label_list[2].extend(_labels)
            elif _dir == "Happy":
                _files = _list_files(_dir, dir)
                _labels = _rem_extn_from_list(_files)
                label_list[3].extend(_labels)
            elif _dir == "Neutral":
                _files = _list_files(_dir, dir)
                _labels = _rem_extn_from_list(_files)
                label_list[4].extend(_labels)
            elif _dir == "Sad":
                _files = _list_files(_dir, dir)
                _labels = _rem_extn_from_list(_files)
                label_list[5].extend(_labels)
            elif _dir == "Surprise":
                _files = _list_files(_dir, dir)
                _labels = _rem_extn_from_list(_files)
                label_list[6].extend(_labels)
            else:
                pass

    os.chdir(ROOT_DIR)

def match_image_with_labels(img_filepath, label_list, img_label_list):

    """
        Match image according to its emotion labels.
    """

    for i in tqdm(range(len(label_list))):
        _label_list = label_list[i]
        
        for j in tqdm(range(len(_label_list))):
            for k in range(len(img_filepath)):
                _img_filepath_split = img_filepath[k].split("\\")
                _imgs_parent_folder = _img_filepath_split[len(_img_filepath_split) - 2]
                
                if _label_list[j] == _imgs_parent_folder:
                    img_label_list[i].append(img_filepath[k])
                    break
    
def arrange_files(dir):

    """
        Using the folder conventions of 0-6 according to basic emotion labels. 
        The images will be organized into these folders.
    """

    _create_folder('labelled_image')
    os.chdir('labelled_image')
    
    _create_folder('0')
    _create_folder('1')
    _create_folder('2')
    _create_folder('3')
    _create_folder('4')
    _create_folder('5')
    _create_folder('6')
    
    os.chdir(ROOT_DIR)

    index = 0

    for i in tqdm(range(len(IMAGE_LABEL_LIST))): 
        filepath_list = IMAGE_LABEL_LIST[i]

        for j in tqdm(range(len(filepath_list))):
            filepath = filepath_list[j]
            if i == 0: # angry
                shutil.copy(filepath, os.path.join(ANGER_FOLDER, \
                            str(index) + '.jpg'))
            if i == 1: # disgust
                shutil.copy(filepath, os.path.join(DISGUST_FOLDER, \
                            str(index) + '.jpg'))
            if i == 2: # fear
                shutil.copy(filepath, os.path.join(FEAR_FOLDER, \
                            str(index) + '.jpg'))
            if i == 3: # happy
                shutil.copy(filepath, os.path.join(HAPPY_FOLDER, \
                            str(index) + '.jpg'))
            if i == 4: # neutral
                shutil.copy(filepath, os.path.join(NEUTRAL_FOLDER, \
                            str(index) + '.jpg'))
            if i == 5: # sad
                shutil.copy(filepath, os.path.join(SAD_FOLDER, \
                            str(index) + '.jpg'))
            if i == 6: # surprise
                shutil.copy(filepath, os.path.join(SURPRISE_FOLDER, \
                        str(index) + '.jpg'))

            index += 1
                    
def main():
    start = time.process_time()
    
    print("Running get_labels from Train data...")
    get_labels(ROOT_DIR + '\\Train_AFEW', LABEL_LIST)
    print("LABEL LIST: ", len(LABEL_LIST))
    print("Running get_labels from Val data...")
    get_labels(ROOT_DIR + '\\Val_AFEW', LABEL_LIST)
    print("LABEL LIST: ", len(LABEL_LIST))

    print("Running get_images() for Train data...")
    get_images(ROOT_DIR + '\\Train_AFEW\\AlignedFaces_LBPTOP_Points\\Faces', IMAGE_FILEPATH)
    print("IMAGE FILEPATH: ", len(IMAGE_FILEPATH))

    print("Running get_images() for Val data...")
    get_images(ROOT_DIR + '\\Val_AFEW\\AlignedFaces_LBPTOP_Points_Val\\Faces', IMAGE_FILEPATH)
    print("IMAGE FILEPATH: ", len(IMAGE_FILEPATH))

    print("Running match_image_with_labels for all data...")
    match_image_with_labels(IMAGE_FILEPATH, LABEL_LIST, IMAGE_LABEL_LIST)
    print("IMAGE_LABEL_LIST: ", (IMAGE_LABEL_LIST))

    print("Running arrange_files...")
    arrange_files(ROOT_DIR)

    print("TIME ELAPSED:", time.process_time() - start)
    
if __name__ == '__main__':

    main()