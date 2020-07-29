from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import scipy.misc
import dlib
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pickle
import logging
from tqdm import tqdm
from keras.optimizers import SGD
from sklearn import preprocessing
from imutils import face_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Convolution2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from numpy.random import seed
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

owd = os.getcwd()

NUM_CORES = 6
NUM_CLASSES = 7
LABELS = ['Neutral', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
BATCH_SIZE = 64
EPOCHS = 100
IMG_ROWS = 48
IMG_COLS = 48
SEED = 8
fit = True
seed(8)

CNN_LR = 0.001
CNN_DECAY = 1e-6
CNN_MOMENTUM = 0.9

def plot_confusion_matrix(cm, classes, normalize=False, \
		title='Confusion matrix', cmap=plt.cm.Blues):
	
	"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
	"""
	
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title, fontsize=24)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
	plt.yticks(tick_marks, classes, fontsize=18)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(
			j, i, format(cm[i, j], fmt), \
			horizontalalignment="center", \
			color="white" if cm[i, j] > thresh else "black", \
			fontsize=18
		)

	plt.ylabel('True label', fontsize=22)
	plt.xlabel('Predicted label', fontsize=22)
	plt.tight_layout()

def plot_classification_loss(history):

	plt.clf()

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(loss) + 1)

	plt.plot(epochs, loss, 'bo', label="Training loss")
	plt.plot(epochs, val_loss, 'b', label="Validation loss")
	plt.title("Training and validation loss")
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

def plot_classification_acc(history):

	plt.clf()

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label="Training accuracy")
	plt.plot(epochs, val_acc, 'b', label="Validation accuracy")
	plt.title("Training and validation accuracy")
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

def model_generate_CNN():

	logging.info("Generating CNN model....")

	model = Sequential()

	# 1st Conv layer
	model.add(Conv2D(32, (3, 3), activation='relu', \
					input_shape=(IMG_ROWS, IMG_COLS, 1)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))

	# 2nd Conv layer
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))

	# 3rd Conv layer
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))

	# Flattening
	model.add(Flatten())

	# Fully connected neural networks
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))

	# Output
	model.add(Dense(NUM_CLASSES, activation='softmax'))

	sgd = SGD(lr=CNN_LR, decay=CNN_DECAY, momentum=CNN_MOMENTUM, nesterov=True)
	model.compile(loss='categorical_crossentropy', \
				optimizer=sgd, \
				metrics=['accuracy']
	)

	#model.summary()

	return model

def test_model():
	model = Sequential()

	model.add(Convolution2D(64, (3, 1), padding='same', input_shape=(48,48,1)))
	model.add(Convolution2D(64, (1, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
	model.add(Dropout(0.25))

	model.add(Convolution2D(128, (3, 1), padding='same'))
	model.add(Convolution2D(128, (1, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
	model.add(Dropout(0.25))

	model.add(Convolution2D(256, (3, 1), padding='same'))
	model.add(Convolution2D(256, (1, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
	model.add(Dropout(0.25))

	model.add(Convolution2D(512, (3, 1), padding='same'))
	model.add(Convolution2D(512, (1, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(7))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

	#model.summary()

	return model

def train_CNN_K_Means(x_train, y_train, model_save_name):

	n_splits = 10
	i = 0
	conf_mat_avg = 0
	cv_scores = []
	conf_mat = []

	#model = model_generate_CNN()
	model = test_model()

	# define 10-fold cross validation test harness
	kfold = MultilabelStratifiedKFold(
					n_splits=n_splits,
					shuffle=True,
					random_state=SEED
	)

	for train_index, test_index in tqdm(kfold.split(x_train, y_train), \
			total=kfold.get_n_splits(), desc="k-fold"):
		i = i + 1

		print ("Running Fold", i, "/", n_splits)
		X_train, X_test = x_train[train_index], \
						x_train[test_index]
		Y_train, Y_test = y_train[train_index], y_train[test_index]

		model_checkpoint = ModelCheckpoint(
						"../model/CNN_model_" + model_save_name + ".h5",
						'val_accuracy',
						verbose=1,
						save_best_only=True
		)

		reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, \
				patience=50, min_lr=0.0001)

		callbacks = [model_checkpoint, reduce_lr]

		history = model.fit(X_train, Y_train, epochs=EPOCHS, verbose=0, \
				batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), \
				callbacks=callbacks)

		model.save_weights('../model/CNN_model_' + model_save_name + '_Last.h5')
		model.load_weights('../model/CNN_model_' + model_save_name + '.h5')

		# Save the model and the weights
		model_json = model.to_json()
		with open("../model/CNN_model_" + model_save_name + ".json", "w") \
			as json_file:
			json_file.write(model_json)

		# save the loss and accuracy data
		f = open('../model/CNN_history_' + model_save_name + '.pckl', 'wb')
		pickle.dump(history.history, f)
		f.close()

		#Evaluate model
		scores = model.evaluate(X_test, Y_test)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cv_scores.append(scores[1] * 100)

		print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

		y_prob  = model.predict(X_test)
		y_classes = y_prob.argmax(axis=-1)
		y_pred = y_classes
		y_true = [0] * len(y_pred)

		for j in range(0, len(Y_test)):
						max_index = np.argmax(Y_test[j])
						y_true[j] = max_index

		conf_mat.append(confusion_matrix(y_true, y_pred, \
						labels=range(NUM_CLASSES)))

	acc_score = np.mean(cv_scores)
	conf_mat = np.array(conf_mat)

	for i in range(len(conf_mat)):
		conf_mat_avg += conf_mat[i]

	conf_mat_avg = conf_mat_avg / 10

	# Plot normalized confusion matrix
	plt.figure(figsize=(10, 8))
	plot_confusion_matrix(
					conf_mat_avg,
					classes=LABELS,
					normalize=True,
					title= "Average Accuracy %: " + str(round(acc_score, 2))
	)

	plt.savefig("../results/train_" + str(model_save_name) + ".png")

def test_CNN_K_Means(x_test, y_test, model_save_name, test_set_name):

	conf_mat = []
	conf_mat_avg = 0
	cv_scores = []
	test_amount = 1

	#model = model_generate_CNN()
	model = test_model()

	model.load_weights('../model/CNN_model_' + model_save_name + '.h5')

	for i in range(test_amount):

		y_pred = model.predict_classes(x_test)
		y_true = [0] * len(y_pred)

		for i in range(0, len(y_test)):
			max_index = np.argmax(y_test[i])
			y_true[i] = max_index

		# Draw the confusion matrix
		conf_mat.append(confusion_matrix(y_true, y_pred, \
			labels=range(NUM_CLASSES)))

		# Evaluate the model on the test set
		scores = model.evaluate(x_test, y_test)
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

		ac = float("{0:.3f}".format(scores[1]*100))
		cv_scores.append(ac)

	acc_score_avg = np.mean(cv_scores)
	conf_mat = np.array(conf_mat)

	for i in range(len(conf_mat)):
		conf_mat_avg += conf_mat[i]

	conf_mat_avg = conf_mat_avg / test_amount

	# Plot normalized confusion matrix
	plt.figure(figsize=(10, 8))
	plot_confusion_matrix(
					conf_mat_avg,
					classes=LABELS,
					normalize=True,
					title= "Average Accuracy %: " + str(round(acc_score_avg, 2))
	)

	plt.savefig("../results/train_" + str(model_save_name) + "-test_" + str(test_set_name) + ".png")

def initialize_tensorflow(num_cores):

	"""
		cpu - gpu configuration
		config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} )
		max: 1 gpu, 56 cpu
	"""

	config = tf.ConfigProto(
					intra_op_parallelism_threads=num_cores,
					inter_op_parallelism_threads=num_cores,
					allow_soft_placement=True,
					device_count = {'CPU' : 1, 'GPU' : 1}
	)

	sess = tf.Session(config=config)
	tf.keras.backend.set_session(sess)

def perform_train_and_test():

	# Training and test data loading
	# # ckplus
	# ckplus_train_images = np.load('../features/ckplus_train_images.npy')
	# ckplus_train_labels = np.load('../features/ckplus_train_labels.npy')
	# ckplus_test_images = np.load('../features/ckplus_test_images.npy')
	# ckplus_test_labels = np.load('../features/ckplus_test_labels.npy')
	# ckplus_full_images = np.load('../features/ckplus_full_images.npy')
	# ckplus_full_labels = np.load('../features/ckplus_full_labels.npy')

	# # ckubd
	# ckubd_train_images = np.load('../features/ckubd_train_images.npy')
	# ckubd_train_labels = np.load('../features/ckubd_train_labels.npy')
	# ckubd_test_images = np.load('../features/ckubd_test_images.npy')
	# ckubd_test_labels = np.load('../features/ckubd_test_labels.npy')
	# ckubd_full_images = np.load('../features/ckubd_full_images.npy')
	# ckubd_full_labels = np.load('../features/ckubd_full_labels.npy')

	# # rafd
	# rafd_train_images = np.load('../features/rafd_train_images.npy')
	# rafd_train_labels = np.load('../features/rafd_train_labels.npy')
	# rafd_test_images = np.load('../features/rafd_test_images.npy')
	# rafd_test_labels = np.load('../features/rafd_test_labels.npy')
	# rafd_full_images = np.load('../features/rafd_full_images.npy')
	# rafd_full_labels = np.load('../features/rafd_full_labels.npy')

	# # kdef
	# kdef_train_images = np.load('../features/kdef_train_images.npy')
	# kdef_train_labels = np.load('../features/kdef_train_labels.npy')
	# kdef_test_images = np.load('../features/kdef_test_images.npy')
	# kdef_test_labels = np.load('../features/kdef_test_labels.npy')
	# kdef_full_images = np.load('../features/kdef_full_images.npy')
	# kdef_full_labels = np.load('../features/kdef_full_labels.npy')

	# # jaffe
	# jaffe_train_images = np.load('../features/jaffe_train_images.npy')
	# jaffe_train_labels = np.load('../features/jaffe_train_labels.npy')
	# jaffe_test_images = np.load('../features/jaffe_test_images.npy')
	# jaffe_test_labels = np.load('../features/jaffe_test_labels.npy')
	# jaffe_full_images = np.load('../features/jaffe_full_images.npy')
	# jaffe_full_labels = np.load('../features/jaffe_full_labels.npy')

	# # sfew2
	# sfew2_train_images = np.load('../features/sfew2_train_images.npy')
	# sfew2_train_labels = np.load('../features/sfew2_train_labels.npy')
	# sfew2_test_images = np.load('../features/sfew2_test_images.npy')
	# sfew2_test_labels = np.load('../features/sfew2_test_labels.npy')
	# sfew2_full_images = np.load('../features/sfew2_full_images.npy')
	# sfew2_full_labels = np.load('../features/sfew2_full_labels.npy')

	# # afew2018
	# afew2018_train_images = np.load('../features/afew2018_train_images.npy')
	# afew2018_train_labels = np.load('../features/afew2018_train_labels.npy')
	# afew2018_test_images = np.load('../features/afew2018_test_images.npy')
	# afew2018_test_labels = np.load('../features/afew2018_test_labels.npy')
	# afew2018_full_images = np.load('../features/afew2018_full_images.npy')
	# afew2018_full_labels = np.load('../features/afew2018_full_labels.npy')

	# # fer2013
	# fer2013_train_images = np.load('../features/fer2013_train_images.npy')
	# fer2013_train_labels = np.load('../features/fer2013_train_labels.npy')
	# fer2013_test_images = np.load('../features/fer2013_test_images.npy')
	# fer2013_test_labels = np.load('../features/fer2013_test_labels.npy')
	# fer2013_full_images = np.load('../features/fer2013_full_images.npy')
	# fer2013_full_labels = np.load('../features/fer2013_full_labels.npy')

	# # expw
	# expw_train_images = np.load('../features/expw_train_images.npy')
	# expw_train_labels = np.load('../features/expw_train_labels.npy')
	# expw_test_images = np.load('../features/expw_test_images.npy')
	# expw_test_labels = np.load('../features/expw_test_labels.npy')
	# expw_full_images = np.load('../features/expw_full_images.npy')
	# expw_full_labels = np.load('../features/expw_full_labels.npy')

	# # # rafdb
	# rafdb_train_images = np.load('../features/rafdb_train_images.npy')
	# rafdb_train_labels = np.load('../features/rafdb_train_labels.npy')
	# rafdb_test_images = np.load('../features/rafdb_test_images.npy')
	# rafdb_test_labels = np.load('../features/rafdb_test_labels.npy')
	# rafdb_full_images = np.load('../features/rafdb_full_images.npy')
	# rafdb_full_labels = np.load('../features/rafdb_full_labels.npy')

	# # # affectnet
	# affectnet_train_images = np.load('../features/affectnet_train_images.npy')
	# affectnet_train_labels = np.load('../features/affectnet_train_labels.npy')
	# affectnet_test_images = np.load('../features/affectnet_test_images.npy')
	# affectnet_test_labels = np.load('../features/affectnet_test_labels.npy')
	# affectnet_full_images = np.load('../features/affectnet_full_images.npy')
	# affectnet_full_labels = np.load('../features/affectnet_full_labels.npy')

	# Perform tests
	# SINGLE
	# train_CNN_K_Means(ckplus_train_images, ckplus_train_labels, "ckplus")
	# test_CNN_K_Means(ckplus_test_images, ckplus_test_labels, "ckplus", "ckplus")
	# test_CNN_K_Means(ckubd_full_images, ckubd_full_labels, "ckplus", "ckubd")
	# test_CNN_K_Means(rafd_full_images, rafd_full_labels, "ckplus", "rafd")
	# test_CNN_K_Means(kdef_full_images, kdef_full_labels, "ckplus", "kdef")
	# test_CNN_K_Means(jaffe_full_images, jaffe_full_labels, "ckplus", "jaffe")
	# test_CNN_K_Means(sfew2_full_images, sfew2_full_labels, "ckplus", "sfew2")
	# test_CNN_K_Means(afew2018_full_images, afew2018_full_labels, "ckplus", "afew2018")
	# test_CNN_K_Means(fer2013_full_images, fer2013_full_labels, "ckplus", "fer2013")
	# test_CNN_K_Means(expw_full_images, expw_full_labels, "ckplus", "expw")
	# test_CNN_K_Means(rafdb_full_images, rafdb_full_labels, "ckplus", "rafdb")
	# test_CNN_K_Means(affectnet_full_images, affectnet_full_labels, "ckplus", "affectnet")

def main():

	initialize_tensorflow(NUM_CORES)

	perform_train_and_test()

if __name__ == '__main__':

	logging.basicConfig(level=logging.INFO)

	main()