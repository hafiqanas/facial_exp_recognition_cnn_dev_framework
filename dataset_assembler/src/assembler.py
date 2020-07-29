from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import os
import cv2
import glob
import keras
import numpy as np
from random import shuffle

owd = os.getcwd()

FULL_DIRECTORY_CKPLUS = '../data/ck+/image/*/*'
FULL_DIRECTORY_CKUBD = '../data/Full_MTCNN_Crop/image/*/*'
FULL_DIRECTORY_RAFDB = '../data/RAFDB/image/*/*'
FULL_DIRECTORY_RAFD = '../data/RAFD_MTCNN_Crop/image/*/*'
FULL_DIRECTORY_KDEF = '../data/KDEF_MTCNN_Crop/image/*/*'
FULL_DIRECTORY_JAFFE = '../data/JAFFE_MTCNN_Crop/image/*/*'
FULL_DIRECTORY_SFEW2 = '../data/sfew2/image/*/*'
FULL_DIRECTORY_FER2013 = '../data/fer2013-reformatted/image/*/*'
FULL_DIRECTORY_RAVDNESS = '../data/RAVDNESS/*/*'
FULL_DIRECTORY_EXPW = '../data/expw-proper/image/*/*'
FULL_DIRECTORY_AFFECTNET = '../data/affectnet/image/*/*'
FULL_DIRECTORY_EMOTIONET = '../data/emotionet/image/*/*'

IMG_ROWS = 48
IMG_COLS = 48
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2
FEATURES_FOLDER = '../features/'

def image_preprocesing(path):

    """
        Converts an img to grayscale and resizes it to 48 x 48 pixels

        path: path to img
    """
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
    
    return img
    
def labels_preprocessing(emotion, classes):

    """
        Converts a label to categorical labels

        emotion: emotion labels (0, 1, 2 ...) (string)
        classes: total number of unique emotion labels (int)
    """

    _emotion = int(emotion)
    emotion = keras.utils.to_categorical(_emotion, classes)
    
    return emotion
    
def img_list_to_npy(list):

    """
        Converts an list of img pixels to numpy

        list: list of img pixels (list)
    """

    _npy = np.array(list)
    _npy = _npy.reshape(_npy.shape[0], IMG_ROWS, IMG_COLS, 1)
    npy = _npy.astype('float32')
    
    return npy
    
def save_as_npy(npy, npy_name):

    """
        Saves a numpy file to the specified features folder

        npy: list of img pixels (list)
        npy_name: name of numpy file (string)
    """

    if not os.path.exists(FEATURES_FOLDER):
        os.makedirs(FEATURES_FOLDER)
    
    np.save(FEATURES_FOLDER + npy_name, npy)

def _train_test_split(train_split, test_split, img_list, label_list):  

    """
        train_split & test_split accepts value 0.0 - 1.0 (0% - 100%)
    """

    img_train = img_list[:int(len(img_list) * train_split)]
    label_train = label_list[:int(len(label_list) * train_split)]
    img_test = img_list[int(len(img_list) * (1.0 - test_split)):]
    label_test = label_list[int(len(label_list) * (1.0 - test_split)):]
    img_full = img_list[:int(len(img_list) * 1.0)]
    label_full = label_list[:int(len(label_list) * 1.0)]

    return img_train, label_train, img_test, label_test, img_full, label_full


def prepare_data(directory, classes, npy_name):
    
    """
        Converts data into a numpy format with the specified train and test
        split for CNN model baseline model training.

        directory: full path to dataset directory
        classes: total number of unique emotion labels (int)
        npy_name: name of numpy file (string)
    """

    _img_list = [[], [], [], [], [], [], []]
    _label_list = [[], [], [], [], [], [], []]
    _x_train, _y_train, _x_test, _y_test, \
        _x_full, _y_full = [], [], [], [], [], []

    for path in glob.glob(directory):
        split_path = path.split(os.sep)
        _img_list[int(split_path[1])].append(path)
        _label_list[int(split_path[1])].append(split_path[1])

    for i in range(len(_img_list)):      
        img_train, label_train, img_test, label_test, img_full, label_full = \
            _train_test_split(TRAIN_SPLIT, TEST_SPLIT, _img_list[i], \
                _label_list[i])
        
        for j in range(len(img_train)):   
            _img_train = image_preprocesing(img_train[j])
            _x_train.append(_img_train)

        for j in range(len(label_train)):   
            _label_train = labels_preprocessing(label_train[j], classes)
            _y_train.append(_label_train)

        for j in range(len(img_test)):   
            _img_test = image_preprocesing(img_test[j])
            _x_test.append(_img_test)

        for j in range(len(label_test)):   
            _label_train = labels_preprocessing(label_test[j], classes)
            _y_test.append(_label_train)

        for j in range(len(img_full)):   
            _img_full = image_preprocesing(img_full[j])
            _x_full.append(_img_full)

        for j in range(len(label_full)):   
            _label_full = labels_preprocessing(label_full[j], classes)
            _y_full.append(_label_full)
        
    _x_train = img_list_to_npy(_x_train)
    _x_test = img_list_to_npy(_x_test)
    _x_full = img_list_to_npy(_x_full)
    
    save_as_npy(_x_train, npy_name + "_train_images")
    save_as_npy(_y_train, npy_name + "_train_labels")
    save_as_npy(_x_test, npy_name + "_test_images")
    save_as_npy(_y_test, npy_name + "_test_labels")
    save_as_npy(_x_full, npy_name + "_full_images")
    save_as_npy(_y_full, npy_name + "_full_labels")
    
def merge_data(npy_files, merged_img_name, merged_label_name):

    """
        Merges two dataset numpy file together.

        npy_files: list of numpy files (list)
        merged_img_name: name of merged image numpy file (string)
        merged_label_name: name of merged image' labels numpy file (string)
    """
    
    merged_images, merged_labels = [], []
    
    for i in range(len(npy_files)):
        if "images" in npy_files[i]:
            print(npy_files[i])
            _images = np.load('../features/' + npy_files[i] + '.npy').tolist() 
            merged_images.extend(_images)
        
        if "labels" in npy_files[i]:
            print(npy_files[i])
            _labels = np.load('../features/' + npy_files[i] + '.npy').tolist() 
            merged_labels.extend(_labels)
    
    save_as_npy(merged_images, merged_img_name + '.npy')
    save_as_npy(merged_labels, merged_label_name + '.npy')

def main():

    """
        Run your sequence of program here.

        Use prepare_data() function to convert single dataset into .npy format.

        Use merge_data() function to merge multiple datasets into a single 
        .npy format.

    """

    prepare_data(FULL_DIRECTORY_CKPLUS, 7, "ckplus")
    prepare_data(FULL_DIRECTORY_CKUBD, 7, "ckubd")
    prepare_data(FULL_DIRECTORY_RAFDB, 7, "rafdb") 
    prepare_data(FULL_DIRECTORY_RAFD, 7, "rafd")
    prepare_data(FULL_DIRECTORY_KDEF, 7, "kdef") 
    prepare_data(FULL_DIRECTORY_JAFFE, 7, "jaffe")
    prepare_data(FULL_DIRECTORY_SFEW2, 7, "sfew2")
    prepare_data(FULL_DIRECTORY_FER2013, 7, "fer2013")
    prepare_data(FULL_DIRECTORY_EXPW, 7, "expw")
    prepare_data(FULL_DIRECTORY_AFFECTNET, 7, "affectnet")
    # prepare_data(FULL_DIRECTORY_EMOTIONET, 6, "emotionet")

    # merge_data(["ckubd_train_images", "expw_train_images", \
    #     "ckubd_train_labels", "expw_train_labels"], \
    #     "merged_ckubd_expw_train_images", "merged_ckubd_expw_train_labels")

    # merge_data(["ckubd_train_images", "fer2013_train_images", "expw_train_images", \
    #     "ckubd_train_labels", "fer2013_train_labels", "expw_train_labels"], \
    #     "merged_ckubd_fer2013_expw_train_images", "merged_ckubd_fer2013_expw_train_labels")
    
if __name__ == '__main__':

    main()