import os

PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATASET_DIR = os.path.join(PACKAGE_ROOT, 'datasets')
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
DATA_FOLDER = os.path.join(DATASET_DIR, 'imageclef-med-concept-detection')

# MODEL PERSISTING
MODEL_NAME = 'cnn_modality_clf'
PIPELINE_NAME = 'cnn_pipe'
CLASSES_NAME = 'classes'
ENCODER_NAME = 'encoder'