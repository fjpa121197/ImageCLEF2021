import os
from numpy.random import seed
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt

from tensorflow.keras.applications import DenseNet121


model = tf.keras.models.load_model('ImageCLEF_Modality_Detection.h5')

test_generator_4 = ImageDataGenerator(rescale=1./255).flow_from_directory(
		directory = '../dataset/ImageCLEF2020_Test_Images/',
        target_size = (256,256),
         shuffle = False, 
         class_mode = 'categorical', 
         batch_size = 32)





predictions = model.predict(test_generator_4)

prediction_classes = np.argmax(predictions, axis = 1) # Returns true for the index with the highest probability

true_classes_4 = test_generator_4.classes # Gets the available classes passed to the generator
class_labels_4 = list(test_generator_4.class_indices.keys()) # Gets the label ffor each class
report_4 = metrics.classification_report(true_classes_4,prediction_classes, target_names = class_labels_4)
print("Based model classification report")
print(report_4)

