"""
    Script to convert emotionet dataset into a 0-6 facial emotion labels folder 
    convention as part of the data preparation process to convert to numpy 
    format or merge with other datasets for CNN baseline model training and 
    testing.

    ## USAGE ##
    
    1. Move this script into the emotionet dataset folder and run the script.
"""

import os
import cv2
import csv
import zipfile
import rarfile
import shutil
import itertools
import xlrd
import requests

ROOT_DIR = os.getcwd()

NEUTRAL_FOLDER = ROOT_DIR + '\\labelled_image\\0'
ANGER_FOLDER = ROOT_DIR + '\\labelled_image\\1'
DISGUST_FOLDER = ROOT_DIR + '\\labelled_image\\2'
FEAR_FOLDER = ROOT_DIR + '\\labelled_image\\3'
HAPPY_FOLDER = ROOT_DIR + '\\labelled_image\\4'
SAD_FOLDER = ROOT_DIR + '\\labelled_image\\5'
SURPRISE_FOLDER = ROOT_DIR + '\\labelled_image\\6'

IMAGE_LABEL_LIST = []

def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def _download_images(img_label_list_fileurl, dir, name):

    """
        download images from the excel file url link.
    """

    img_data = requests.get(img_label_list_fileurl).content
        
    with open(dir + '\\' + str(name) + ".jpg", 'wb') as handler:
        handler.write(img_data)
    
def get_images_with_labels(dir, img_label_list):

    """
        Go through the excel file to get all row and col values.
    """
    
    workbook = xlrd.open_workbook(dir + "\\URLsWithEmotionCat_aws.xlsx")
    sheet = workbook.sheet_by_index(0)

    for rowx in range(sheet.nrows):
        values = sheet.row_values(rowx)
        img_label_list.append(values)
    
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

    count = 0

    for i in range(len(IMAGE_LABEL_LIST)):
        fileurl = IMAGE_LABEL_LIST[i][0]
        # neutral_label = IMAGE_LABEL_LIST[i][1]
        happy_label = IMAGE_LABEL_LIST[i][13]
        sad_label = IMAGE_LABEL_LIST[i][14]
        surprise_label = IMAGE_LABEL_LIST[i][17]
        fear_label = IMAGE_LABEL_LIST[i][8]
        disgust_label = IMAGE_LABEL_LIST[i][7]
        anger_label = IMAGE_LABEL_LIST[i][4]

        # if label == 0: # neutral
        #     download_images(fileurl, NEUTRAL_FOLDER, count)
        if happy_label == 1: # happy
            _download_images(fileurl, HAPPY_FOLDER, count)
        if sad_label == 1: # sad
            _download_images(fileurl, SAD_FOLDER, count)
        if surprise_label == 1: # surprise
            _download_images(fileurl, SURPRISE_FOLDER, count)
        if fear_label == 1: # fear
            _download_images(fileurl, FEAR_FOLDER, count)
        if disgust_label == 1: # disgust
            _download_images(fileurl, DISGUST_FOLDER, count)
        if anger_label == 1: # anger
            _download_images(fileurl, ANGER_FOLDER, count)

        count += 1
                    
def main():
    print("Running get_images()...")
    get_images_with_labels(ROOT_DIR, IMAGE_LABEL_LIST)
    print("Running arrange_files()...")
    arrange_files(ROOT_DIR)
    
if __name__ == '__main__':

    main()