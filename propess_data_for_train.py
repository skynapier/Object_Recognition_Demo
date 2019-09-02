from tensorflow.python.keras.preprocessing.image import img_to_array

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import cv2
import os
import argparse
import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

def extract_features(img_list):
    feature_list = []
    for i in range(len(img_list)):
        feature = []
        for channel in range(0,3):
            img = img_list[i][:, :, channel]
            mean, std = float('%.4f'%np.mean(img)),float('%.4f'%np.std(img))
            feature.append(mean)
            feature.append(std)
        feature_list.append(feature)

    return feature_list


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--train_data_dir", default = "Train_data",
                      help = "path to train_data_dir")
    args = vars(args.parse_args())
    return args


def load_images(test_data_dir, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(test_data_dir)))
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)

        #pre-processing
        img1 = np.rot90(image)
        img2 = np.rot90(image,3)

        crop_image = image[30:240, 30:240]
        image = cv2.resize(crop_image, image_size)
        image = img_to_array(image)
        images_data.append(image)

        crop_image1 = img1[30:240, 30:240]
        image1 = cv2.resize(crop_image1, image_size)
        image1 = img_to_array(image1)
        images_data.append(image1)

        crop_image2 = img2[30:240, 30:240]
        image2 = cv2.resize(crop_image2, image_size)
        image2 = img_to_array(image2)
        images_data.append(image2)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        labels.append(label)
        labels.append(label)


    return images_data, sorted(labels)


def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_test = np.array(images, dtype="float")
    y_test = np.array(labels)

    #Binarize the labels
    # convert class vectors to binary class matrices
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return X_test, y_test


def get_data_for_train():
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["train_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    # Load images
    images, labels = load_images(test_data_dir, image_size)

    data, labels = convert_img_to_array(images, labels)

    X_train,X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=309)

    return (X_train, y_train), (X_test, y_test)

def get_data_for_MLP_train():

    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["train_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    # Load images
    images, labels = load_images(test_data_dir, image_size)

    data = extract_features(images)
    return data,labels
