"""
    Script to convert affectnet dataset into a 0-6 facial emotion labels folder 
    convention as part of the data preparation process to convert to numpy 
    format or merge with other datasets for CNN baseline model training and 
    testing.

    ## USAGE ##
    
    1. Within affectnet dataset folder, extract the following zip files 
    ("Automatically_annotated_compressed", "Automatically_annotated_file_list", 
    "Manually_annotated_compressed", "Manually_Annotated_file_lists") and its
    content.
    3. Move this script into the affectnet folder and run the script.
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
LABEL_LIST_COPY = []
IMAGE_LABEL_LIST = []
LABELLED_CROPPED_IMAGE_NAME = []

def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def _save_bounding_box_img(path, name, img):
    cv2.imwrite(os.path.join(path , name), img)

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
            image_file_name = str(filepath.split(os.sep)[path_list_len])
            
            raw_img_filepath.append(filepath)
            raw_img_names.append(image_name)
            raw_img_filename.append(image_file_name)

    os.chdir(ROOT_DIR)

def get_cropped_images(dir, cropped_img_filepath, cropped_img_names):

    """
        Go through specified directory to look for all images' filepath.
    """

    os.chdir(dir)
    
    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            cropped_img_filepath.append(filepath)
            cropped_img_names.append(file)

    os.chdir(ROOT_DIR)

def get_labels(dir, which, list, label_list):

    """
        Go through all .csv files to look for all images labels and bounding 
        boxes.
    """

    count = 0

    if which == "manual":
        with open(dir + '\\Manually_Annotated_file_lists\\training.csv') as inf:
            reader = csv.reader(inf)
            for row in reader:
                if row[0] in list:
                    count += 1
                    label_list.append(row)
        with open(dir + '\\Manually_Annotated_file_lists\\validation.csv') as inf:
            reader = csv.reader(inf)
            for row in reader:
                if row[0] in list:
                    count += 1
                    label_list.append(row)

    if which == "automatic":
        with open(dir + '\\Automatically_annotated_file_list\\automatically_annotated.csv') as inf:
            reader = csv.reader(inf)
            for row in reader:
                if row[0] in list:
                    count += 1
                    label_list.append(row)

def extract_bounding_box(path, x, y, width, height):

    """
        Remove areas outside of the defined "face box" given in the label files.
    """

    box_img = None

    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        box_img = img[x:width, y:height]

    except cv2.error as e:
        print(e)
        pass

    return box_img

def get_face_image(raw_img_filepath, raw_img_names, raw_img_filename, \
        label_list, labelled_cropped_img_names):

    """
        Extract the face image using the given label files.
    """

    _create_folder('cropped_images')

    for i in tqdm(range(len(raw_img_names))):
        for j in range(len(label_list)):
            if raw_img_names[i] == label_list[j][0]:
                face_img = extract_bounding_box(raw_img_filepath[i], \
                    int(label_list[j][1]), int(label_list[j][2]), \
                        int(label_list[j][3]), int(label_list[j][4]))
                if face_img is not None:
                    _save_bounding_box_img(CROPPED_IMG_FOLDER, \
                        str(raw_img_filename[i]), face_img)

def match_image_with_labels(cropped_img_names, cropped_img_filepath, \
        label_list, img_label_list):

    """
        Match image according to its emotion labels.
    """

    for i in tqdm(range(len(cropped_img_names))):
        img_name, extension = os.path.splitext(cropped_img_names[i])
        for j in range(len(label_list)):
            subdir_path, extension = os.path.splitext(label_list[j][0])
            if str(img_name) in str(subdir_path):
                img_label_list.append([cropped_img_filepath[i], \
                        int(label_list[j][6])])
    
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
        filename = CROPPED_IMAGE_NAME[i] 
        filepath = IMAGE_LABEL_LIST[i][0]
        label = IMAGE_LABEL_LIST[i][1]
        if label == 0: # neutral
            shutil.copy(filepath, os.path.join(NEUTRAL_FOLDER, \
                        filename))
        if label == 1: # happy
            shutil.copy(filepath, os.path.join(HAPPY_FOLDER, \
                        filename))
        if label == 2: # sad
            shutil.copy(filepath, os.path.join(SAD_FOLDER, \
                        filename))
        if label == 3: # surprise
            shutil.copy(filepath, os.path.join(SURPRISE_FOLDER, \
                        filename))
        if label == 4: # fear
            shutil.copy(filepath, os.path.join(FEAR_FOLDER, \
                        filename))
        if label == 5: # disgust
            shutil.copy(filepath, os.path.join(DISGUST_FOLDER, \
                        filename))
        if label == 6: # anger
            shutil.copy(filepath, os.path.join(ANGER_FOLDER, \
                    filename))
                    
def main():
    start = time.process_time()
    
    print("Running get_images() for manually annotated..")
    get_images(ROOT_DIR + '\\Manually_annotated_compressed\\', IMAGE_FILEPATH, IMAGE_NAME, IMAGE_FILENAME)
    print("Running get_labels() for manually annotated...")
    get_labels(ROOT_DIR, "manual", IMAGE_NAME, LABEL_LIST)

    LABEL_LIST_COPY.extend(LABEL_LIST)

    print("man img: ", len(IMAGE_NAME))
    print("man label: ", len(LABEL_LIST))
    print("Running get_face_image() for manually annotated...")
    get_face_image(IMAGE_FILEPATH, IMAGE_NAME, IMAGE_FILENAME, LABEL_LIST, LABELLED_CROPPED_IMAGE_NAME)

    IMAGE_FILEPATH.clear()
    IMAGE_FILENAME.clear()
    IMAGE_NAME.clear()
    LABEL_LIST.clear()
    LABELLED_CROPPED_IMAGE_NAME.clear()

    print("Running get_images() for automatically annotated...")
    get_images(ROOT_DIR + '\\Automatically_annotated_compressed\\', IMAGE_FILEPATH, IMAGE_NAME, IMAGE_FILENAME)
    print("Running get_labels() for automatically annotated...")
    get_labels(ROOT_DIR, "automatic", IMAGE_NAME, LABEL_LIST)

    LABEL_LIST_COPY.extend(LABEL_LIST)

    print("auto img: ", len(IMAGE_NAME))
    print("auto label: ", len(LABEL_LIST))
    print("Running get_face_image() for automatically annotated...")
    get_face_image(IMAGE_FILEPATH, IMAGE_NAME, IMAGE_FILENAME, LABEL_LIST, LABELLED_CROPPED_IMAGE_NAME)

    IMAGE_FILEPATH.clear()
    IMAGE_FILENAME.clear()
    IMAGE_NAME.clear()
    LABEL_LIST.clear()
    LABELLED_CROPPED_IMAGE_NAME.clear()

    print("LABEL_LIST_COPY: ", len(LABEL_LIST_COPY))

    print("Running get_croppped_images()...")
    get_cropped_images(CROPPED_IMG_FOLDER, CROPPED_IMAGE_FILEPATH, CROPPED_IMAGE_NAME)
    print("Running match_image_with_labels()...")
    match_image_with_labels(CROPPED_IMAGE_NAME, CROPPED_IMAGE_FILEPATH, LABEL_LIST_COPY, IMAGE_LABEL_LIST)

    print("Running arrange_files()...")
    arrange_files(ROOT_DIR)

    print("TIME ELAPSED:", time.process_time() - start)
    
if __name__ == '__main__':

    main()