from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
import numpy as np
import pandas as pd
import csv
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import tensorflow
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
import tensorflow as tf


dic = pd.read_pickle(r'mlb_dic_full.pkl')

model = tf.keras.models.load_model('model3b.h5')


validation_images_path = './ImageCLEF2021_ConceptDetection_Validation-Set/Validation-Images/'

for image in os.listdir(validation_images_path):

    path_to_image = os.path.join(validation_images_path, image)
    img = tensorflow.keras.preprocessing.image.load_img(path = path_to_image, target_size = (64,64,3)) # Load actual image
    img = tensorflow.keras.preprocessing.image.img_to_array(img) # Transform image to array of shape (input_shape)
    img = tensorflow.keras.applications.densenet.preprocess_input(img) # This preprocess_input normalizes the pixel values based on imagenet dataset and rescale to a 0-1 values.
    pred = model.predict(np.expand_dims(img, axis=0))
    print(len(pred))