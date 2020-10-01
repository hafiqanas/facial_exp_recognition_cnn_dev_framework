"""
    Script to convert affwild2 dataset into a 0-6 facial emotion labels folder
    convention as part of the data preparation process to convert to numpy
    format or merge with other datasets for CNN baseline model training and
    testing.

    ## USAGE ##

    1. Within affwild2 dataset folder, extract the following zip files
    ("Automatically_annotated_compressed", "Automatically_annotated_file_list",
    "Manually_annotated_compressed", "Manually_Annotated_file_lists") and its
    content.
    3. Move this script into the affwild2 folder and run the script.
"""

import os
import cv2
import csv
import zipfile
import rarfile
import shutil
import itertools
import time
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

IMAGE_FILEPATH = []
IMAGE_NAME = []
IMAGE_FILENAME = []
CROPPED_IMAGE_NAME = []
CROPPED_IMAGE_FILEPATH = []
LABEL_LIST = []
TESTING_FILE_LIST = []
TESTING_IMG_COUNT_LIST = []
IMAGE_LABEL_LIST = []
LABELLED_CROPPED_IMAGE_NAME = []

def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def get_images(dir, raw_img_filepath, raw_img_names, raw_img_filename):

    """
        Go through specified directory to look for all images' filepath.
    """

    os.chdir(dir)

    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            path_list_len = len(filepath.split(os.sep)) - 1
            image_name = str(filepath.split(os.sep)[path_list_len - 1]) + '/' \
                + file
            image_name_no_ext = os.path.splitext(image_name)[0]
            image_file_name = str(filepath.split(os.sep)[path_list_len])

            raw_img_filepath.append(filepath)
            raw_img_names.append(image_name_no_ext)
            raw_img_filename.append(image_file_name)

    os.chdir(ROOT_DIR)

def get_labels(dir, label_list):

    """
        Go through all .txt files to look for all images labels and bounding
        boxes.
    """

    os.chdir(dir)
    
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            count = 1

            filepath = subdir + os.sep + file

            with open(filepath, "r") as fd:
                next(fd)
                for line in fd:
                    filepath_split = filepath.split("\\")
                    _folder_name = filepath_split[len(filepath_split) - 1]
                    folder_name_no_ext = os.path.splitext(_folder_name)[0]
                    line = line.strip()
                    label_list.append([str(folder_name_no_ext) + "/" \
                        + str(count).zfill(5), line])

                    count += 1

    os.chdir(ROOT_DIR)

def match_image_with_labels(img_name_list, img_filepath_list, \
        label_list, img_label_list):

    """
        Match image according to its emotion labels.
    """

    for i in tqdm(range(len(img_name_list))):
        #img_name = os.path.splitext(str(img_name_list[i]))[0]
        #img_name = str(img_name).split('/')[1]
        img_name = img_name_list[i]
        for j in range(len(label_list)):
            label_name = label_list[j][0]
            label_cat = label_list[j][1]
            if img_name == label_name:
                new_img_name = img_name_list[i].replace('/', '_') + ".jpg"
                img_label_list.append([new_img_name, img_filepath_list[i], int(label_cat)])
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

    for i in tqdm(range(len(IMAGE_LABEL_LIST))):
        filename = IMAGE_LABEL_LIST[i][0]
        filepath = IMAGE_LABEL_LIST[i][1]
        label = IMAGE_LABEL_LIST[i][2]
        if label == 0: # neutral
            shutil.copy(filepath, os.path.join(NEUTRAL_FOLDER, \
                        filename))
        if label == 1: # angry
            shutil.copy(filepath, os.path.join(ANGER_FOLDER, \
                        filename))
        if label == 2: # disgust
            shutil.copy(filepath, os.path.join(DISGUST_FOLDER, \
                        filename))
        if label == 3: # fear
            shutil.copy(filepath, os.path.join(FEAR_FOLDER, \
                        filename))
        if label == 4: # happy
            shutil.copy(filepath, os.path.join(HAPPY_FOLDER, \
                        filename))
        if label == 5: # sad
            shutil.copy(filepath, os.path.join(SAD_FOLDER, \
                        filename))
        if label == 6: # surprise
            shutil.copy(filepath, os.path.join(SURPRISE_FOLDER, \
                    filename))

def main():
    start = time.process_time()

    print("Running get_images()..")

    get_images(ROOT_DIR + '\\cropped_aligned\\', IMAGE_FILEPATH, IMAGE_NAME, IMAGE_FILENAME)
    print("Running get_labels() for training set...")
    get_labels(ROOT_DIR + '\\annotation\\EXPR_Set\\Training_Set', LABEL_LIST)
    print("Running get_labels() for Validation set...")
    get_labels(ROOT_DIR + '\\annotation\\EXPR_Set\\Validation_Set', LABEL_LIST)
    
    print("img name: ", len(IMAGE_NAME))
    # print("img: ", IMAGE_NAME)
    print("img filepath: ", len(IMAGE_FILEPATH))
    # print("img: ", IMAGE_FILEPATH)
    print("label: ", len(LABEL_LIST))
    # print(LABEL_LIST)
    
    print("Running match_image_with_labels()...")
    match_image_with_labels(IMAGE_NAME, IMAGE_FILEPATH, LABEL_LIST, IMAGE_LABEL_LIST)
    
    print("img label: ", len(IMAGE_LABEL_LIST))
    
    print("Running arrange_files()...")
    arrange_files(ROOT_DIR)

    print("TIME ELAPSED:", time.process_time() - start)

if __name__ == '__main__':

    main()
