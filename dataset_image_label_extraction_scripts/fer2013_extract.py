"""
    Script to convert fer2013 dataset into a 0-6 facial emotion labels folder 
    convention as part of the data preparation process to convert to numpy 
    format or merge with other datasets for CNN baseline model training and 
    testing.

    ## USAGE ##
    
    1. Move this script into the fer2013 dataset folder and run the script.
"""

import os
import cv2
import shutil

ROOT_DIR = os.getcwd()

NEUTRAL_FOLDER = ROOT_DIR + '\\labelled_image\\0'
ANGER_FOLDER = ROOT_DIR + '\\labelled_image\\1'
DISGUST_FOLDER = ROOT_DIR + '\\labelled_image\\2'
FEAR_FOLDER = ROOT_DIR + '\\labelled_image\\3'
HAPPY_FOLDER = ROOT_DIR + '\\labelled_image\\4'
SAD_FOLDER = ROOT_DIR + '\\labelled_image\\5'
SURPRISE_FOLDER = ROOT_DIR + '\\labelled_image\\6'

def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

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

    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            if filepath.endswith(".png"):
                if "neutral" in filepath:
                    shutil.copy(filepath, os.path.join(NEUTRAL_FOLDER, \
                        str(index) + ".png"))
                if "angry" in filepath:
                    shutil.copy(filepath, os.path.join(ANGER_FOLDER, \
                        str(index) + ".png"))
                if "disgust" in filepath:
                    shutil.copy(filepath, os.path.join(DISGUST_FOLDER, \
                        str(index) + ".png"))
                if "fear" in filepath:
                    shutil.copy(filepath, os.path.join(FEAR_FOLDER, \
                        str(index) + ".png"))
                if "happy" in filepath:
                    shutil.copy(filepath, os.path.join(HAPPY_FOLDER, \
                        str(index) + ".png"))
                if "sad" in filepath:
                    shutil.copy(filepath, os.path.join(SAD_FOLDER, \
                        str(index) + ".png"))
                if "surprise" in filepath:
                    shutil.copy(filepath, os.path.join(SURPRISE_FOLDER, \
                        str(index) + ".png"))

                index += 1

def main():
    arrange_files(ROOT_DIR + '\\train\\image')

if __name__ == '__main__':

    main()