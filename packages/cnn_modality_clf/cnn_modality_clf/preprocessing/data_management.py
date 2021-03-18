import os
import logging
from glob import glob
from pathlib import Path

import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import joblib
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from cnn_modality_clf.config import config

_logger = logging.getLogger(__name__)

def image_generator(data_path: str, predict_flag: bool = False, 
                    image_size: int = 256, batch_size: int, subset: str) :

    if predict_flag == True:
        generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory = data_path,
                                                                        target_size=(image_size, image_size), 
                                                                        batch_size = batch_size, 
                                                                        subset= subset, 
                                                                        seed = 14, 
                                                                        class_mode = 'categorical')
    else:
        generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(directory = data_path,
                                                                                                target_size=(image_size, image_size), 
                                                                                                batch_size = batch_size, 
                                                                                                subset= subset, 
                                                                                                seed = 14, 
                                                                                                class_mode = 'categorical')

    return generator