import os
import logging
from glob import glob
from pathlib import Path

import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

#from cnn_modality_clf.config import config

_logger = logging.getLogger(__name__)

def load_images_paths(data_folder: str) -> pd.DataFrame:

    images_df = []

    # navigate within each folder
    for class_folder_name in os.listdir(data_folder):
        class_folder_path = os.path.join(data_folder, class_folder_name)
        # collect every image path
        for image_path in glob(os.path.join(class_folder_path, "*.jpg")):
            tmp = pd.DataFrame([image_path, class_folder_name]).T
            images_df.append(tmp)

    # concatenate the final df
    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    images_df.columns = ['image', 'target']

    return images_df

def get_train_test_target(df: pd.DataFrame):

    X_train, X_test, y_train, y_test = train_test_split(df['image'],
                                                        df['target'],
                                                        test_size=0.20,
                                                        random_state=101)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test