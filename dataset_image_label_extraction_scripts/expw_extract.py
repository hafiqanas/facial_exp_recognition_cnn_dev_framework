"""
    Script to convert expw dataset into a 0-6 facial emotion labels folder 
    convention as part of the data preparation process to convert to numpy 
    format or merge with other datasets for CNN baseline model training and 
    testing.

    ## USAGE ##
    
    1. Move this script into the expw dataset folder and run the script.
"""

import os
import cv2
import zipfile
import shutil

ROOT_DIR = os.getcwd()

NEUTRAL_FOLDER = ROOT_DIR + '\\labelled_images\\0'
ANGER_FOLDER = ROOT_DIR + '\\labelled_images\\1'
DISGUST_FOLDER = ROOT_DIR + '\\labelled_images\\2'
FEAR_FOLDER = ROOT_DIR + '\\labelled_images\\3'
HAPPY_FOLDER = ROOT_DIR + '\\labelled_images\\4'
SAD_FOLDER = ROOT_DIR + '\\labelled_images\\5'
SURPRISE_FOLDER = ROOT_DIR + '\\labelled_images\\6'

CROPPED_IMG_FOLDER = ROOT_DIR + '\\cropped_images'

IMAGE_NAME = []
IMAGE_FILEPATH = []
CROPPED_IMAGE_NAME = []
CROPPED_IMAGE_FILEPATH = []
LABEL_LIST = []
IMAGE_LABEL_LIST = []

def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def _save_bounding_box_img(path, name, img):
    cv2.imwrite(os.path.join(path , name), img)

def get_images(dir, image_name, image_filepath):

    """
        Go through specified directory to look for all images' filepath.
    """

    os.chdir(dir)
    
    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg"):
                image_name.append(file)
                image_filepath.append(filepath)

    os.chdir(ROOT_DIR)

def get_cropped_images(dir, cropped_image_name, cropped_image_filepath):

    """
        Go through specified directory to look for all images' filepath.
    """

    os.chdir(dir)
    
    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg"):
                cropped_image_name.append(file)
                cropped_image_filepath.append(filepath)

    os.chdir(ROOT_DIR)

def get_labels(dir, label_list):

    """
        Go through specified directory to look for all image labels
    """

    os.chdir(dir)
    
    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            if filepath.endswith(".lst"):
                with open(filepath) as f:
                    for line in f:
                        _line = line.split()
                        _label = _line
                        label_list.append(_label)

    os.chdir(ROOT_DIR)

def extract_bounding_box(path, top, left, right, bottom):

    """
        Remove areas outside of the defined "face box" given in the label files.
    """

    box_img = None

    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        box_img = img[top:bottom, left:right]

    except cv2.error as e:
        print(e)
        pass

    return box_img

def get_face_image(img_list, label_list, image_filepath, cropped_img_folder):

    """
        Extract the face image using the given label files.
    """

    _create_folder('cropped_images')

    for i in range(len(img_list)):
        for j in range(len(label_list)):
            if img_list[i] == label_list[j][0]:
                face_img = extract_bounding_box(image_filepath[i], \
                        int(label_list[j][2]), int(label_list[j][3]), \
                            int(label_list[j][4]), int(label_list[j][5]))
                if face_img is not None:
                    _save_bounding_box_img(cropped_img_folder, \
                            str(img_list[i]), face_img)

def match_image_with_labels(img_list, label_list, image_label_list, \
        cropped_image_filepath):

    """
        Match image according to its emotion labels.
    """

    for i in range(len(img_list)):
        for j in range(len(label_list)):
            if img_list[i] == label_list[j][0]:
                image_label_list.append([cropped_image_filepath[i], \
                        int(label_list[j][7])])
    
def arrange_files(dir):

    """
        Using the folder conventions of 0-6 according to basic emotion labels. 
        The images will be organized into these folders.
    """
    
    _create_folder('labelled_images')
    os.chdir('labelled_images')
    
    _create_folder('0')
    _create_folder('1')
    _create_folder('2')
    _create_folder('3')
    _create_folder('4')
    _create_folder('5')
    _create_folder('6')
    
    os.chdir(ROOT_DIR)
    
    index = 1

    for i in range(len(IMAGE_NAME)):
        try:
            filename = CROPPED_IMAGE_NAME[i]
            filepath = IMAGE_LABEL_LIST[i][0]
            label = IMAGE_LABEL_LIST[i][1]
            if label == 0: # angry
                shutil.copy(filepath, os.path.join(ANGER_FOLDER, \
                            filename))
            if label == 1: # disgust
                shutil.copy(filepath, os.path.join(DISGUST_FOLDER, \
                            filename))
            if label == 2: # fear
                shutil.copy(filepath, os.path.join(FEAR_FOLDER, \
                            filename))
            if label == 3: # happy
                shutil.copy(filepath, os.path.join(HAPPY_FOLDER, \
                            filename))
            if label == 4: # sad
                shutil.copy(filepath, os.path.join(SAD_FOLDER, \
                            filename))
            if label == 5: # surprise
                shutil.copy(filepath, os.path.join(SURPRISE_FOLDER, \
                            filename))
            if label == 6: # neutral
                shutil.copy(filepath, os.path.join(NEUTRAL_FOLDER, \
                        filename))
            index += 1

        except IndexError as e:
            pass
                    
def main():
    get_images(ROOT_DIR + '\\image', IMAGE_NAME, IMAGE_FILEPATH)
    get_labels(ROOT_DIR + '\\label', LABEL_LIST)
    get_face_image(IMAGE_NAME, LABEL_LIST, IMAGE_FILEPATH, CROPPED_IMG_FOLDER)
    get_cropped_images(CROPPED_IMG_FOLDER, CROPPED_IMAGE_NAME, CROPPED_IMAGE_FILEPATH)
    match_image_with_labels(CROPPED_IMAGE_NAME, LABEL_LIST, IMAGE_LABEL_LIST, CROPPED_IMAGE_FILEPATH)
    arrange_files(ROOT_DIR)
    
if __name__ == '__main__':

    main()