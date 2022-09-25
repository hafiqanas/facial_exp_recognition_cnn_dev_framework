# Facial Expression Recognition (FER) development framework

This repository contains three main scripts that completes the pipeline for CNN-based FER model training and testing. This framework was used to train our FER model for the [FG-2020 Competition: ABAW Track 2 Expression Challenge](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/).

FER is a recognition task that classifies the expression of face images into different classes such as happy, sad, anger, fear and so on. This development framework is based on the classical model of FER which classifies basic expressions which is different than the more recent Valence-Arousal model. 

If you have found this repository useful or have used this repository in any of your scientific work, please consider citing my work using this [BibTeX Citation](#bibtex-citation).

# Table of contents
* Repository folders
  * [Dataset image and labels extractor](#dataset-image-and-label-extractor)
  * [Dataset train and test set assembler](#dataset-train-and-test-set-assembler)
  * [Model training and testing scripts](#model-training-and-testing-scripts)
* [Getting started](#getting-started)
* [BibTeX Citation](#bibtex-citation)
* [Acknowledgments](#acknowledgments)

## Dataset image and label extractor
This script extracts images into six or seven different folders which corresponds to the facial expression labels given by the datasets. These are the list of datasets that can be extracted with our script:
* [AFEW2018](https://cs.anu.edu.au/few/AFEW.html)
* [AffectNet](http://mohammadmahoor.com/affectnet/)
* [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
* [CK+](http://www.jeffcohn.net/Resources/)
* [EmotionNet](https://cbcsl.ece.ohio-state.edu/downloads.html)
* [ExpW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html)
* [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
* [RAF-DB](https://rafd.socsci.ru.nl/RaFD2/RaFD?p=main)
* [SFEW2](https://cs.anu.edu.au/few/AFEW.html)

## Dataset train and test set assembler
This script reformats the extracted dataset into a train and test split in ```.npy``` format which are generated in the ```/features``` folder. An example is given in the folder where ```CK+UBD``` dataset was extracted from the previous step and assembled.

## Model training and testing
This script completes the pipeline process which trains a CNN model using the assembled train and test set and saves the model in ```.h5``` and ```.json``` format in the ```/models``` folder. Classification accuracy and loss can also be visualized by a generating a graph using this script.

## Getting started
The whole framework was built on ```Python 3.7.6``` and the entire process pipeline is as follows: 

1. The first step is to download the supported datasets and extract (unzip) them. Then, place this script in the dataset folder and run it. This will generated a folder containing six or seven subfolders that correspond to the different expression classes available to the extracted dataset.
E.g AFEW2018 dataset
```
python afew2018_extract.py
```

2. Run the assembler script pointing the directory location to the extracted dataset. This will generate a ```.npy``` features file which are the train and test split data.
```
python assembler.py
```

3. Run the training or testing pointing the feature directory location to the location of the generated ```.npy``` features file. This will generate a trained model in ```.h5``` and ```.json``` format.
```
python baseline_model.py
```

## BibTeX Citation
If you have used this repository in any of your scientific work, please consider citing my work:
```
@article{anas2020deep,
  title={Deep convolutional neural network based facial expression recognition in the wild},
  author={Anas, Hafiq and Rehman, Bacha and Ong, Wee Hong},
  journal={arXiv preprint arXiv:2010.01301},
  year={2020}
}
```

## Acknowledgments

* Thank you [Robolab@UBD](https://ailab.space/) for lending their lab facilities. 
