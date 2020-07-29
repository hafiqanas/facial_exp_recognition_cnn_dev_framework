"""
    Script to convert rafdb dataset into a 0-6 facial emotion labels folder 
    convention as part of the data preparation process to convert to numpy 
    format or merge with other datasets for CNN baseline model training and 
    testing.

    ## USAGE ##
    
    1. Within rafdb dataset folder, extract the following zip files 
    ("aligned.zip") from the /basic/image folder.
    2. Move this script into the /basic folder and run the script.
"""

import os
import cv2
import zipfile
import shutil

ROOT_DIR = os.getcwd()

NEUTRAL_FOLDER = ROOT_DIR + '\\labelled_image\\0'
ANGER_FOLDER = ROOT_DIR + '\\labelled_image\\1'
DISGUST_FOLDER = ROOT_DIR + '\\labelled_image\\2'
FEAR_FOLDER = ROOT_DIR + '\\labelled_image\\3'
HAPPY_FOLDER = ROOT_DIR + '\\labelled_image\\4'
SAD_FOLDER = ROOT_DIR + '\\labelled_image\\5'
SURPRISE_FOLDER = ROOT_DIR + '\\labelled_image\\6'

IMAGE_FILEPATH = []
IMAGE_ORIGINAL_FILEPATH = []
LABEL_LIST = []

def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def _fix_labels(label):

    """
        Convert rafdb's different emotion to number mapping labels of 1-7 
        into 0-6.
    """

    if label is 0:
        return 6

    if label is 1:
        return 3

    if label is 2:
        return 2

    if label is 3:
        return 4

    if label is 4:
        return 5

    if label is 5:
        return 1

    if label is 6:
        return 0
    
def get_images(dir, image_original_filepath, image_filepath):

    """
        Go through specified directory to look for all images' filepath.
    """

    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg"):
                image_original_filepath.append(filepath)

                new_file = file
                new_file = new_file.replace('_aligned','')
                new_filepath = subdir + os.sep + new_file

                image_filepath.append(new_filepath)

def get_labels(dir, label_list):

    """
        Go through specified directory to look for all image labels
    """

    with open(dir, 'r') as f:
        for line in f:
            line_split = line.split()
            emotion = int(line_split[1]) - 1
            emotion = _fix_labels(emotion)
            label = [line_split[0], str(emotion)]

            label_list.append(label)
    
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

    index = 1

    for i in range(len(IMAGE_FILEPATH)):
        filepath_split = IMAGE_FILEPATH[i].split(os.sep)
        img_filename = filepath_split[len(filepath_split)-1]
        
        for j in range(len(LABEL_LIST)):
            label_img_filename = LABEL_LIST[j][0]
            label = LABEL_LIST[j][1]
            
            if img_filename == label_img_filename:
                if label == '0':
                    shutil.copy(IMAGE_ORIGINAL_FILEPATH[i], os.path.join(NEUTRAL_FOLDER, \
                str(index) + ".jpg"))
                if label == '1':
                    shutil.copy(IMAGE_ORIGINAL_FILEPATH[i], os.path.join(ANGER_FOLDER, \
                str(index) + ".jpg"))
                if label == '2':
                    shutil.copy(IMAGE_ORIGINAL_FILEPATH[i], os.path.join(DISGUST_FOLDER, \
                str(index) + ".jpg"))
                if label == '3':
                    shutil.copy(IMAGE_ORIGINAL_FILEPATH[i], os.path.join(FEAR_FOLDER, \
                str(index) + ".jpg"))
                if label == '4':
                    shutil.copy(IMAGE_ORIGINAL_FILEPATH[i], os.path.join(HAPPY_FOLDER, \
                str(index) + ".jpg"))
                if label == '5':
                    shutil.copy(IMAGE_ORIGINAL_FILEPATH[i], os.path.join(SAD_FOLDER, \
                str(index) + ".jpg"))
                if label == '6':
                    shutil.copy(IMAGE_ORIGINAL_FILEPATH[i], os.path.join(SURPRISE_FOLDER, \
                str(index) + ".jpg"))
            
            index += 1
                    
def main():
    get_images(ROOT_DIR + '\\Image\\aligned', image_original_filepath, image_filepath)
    get_labels(ROOT_DIR + '\\EmoLabel\\list_patition_label.txt', LABEL_LIST)
    arrange_files(ROOT_DIR)
    
if __name__ == '__main__':

    main()