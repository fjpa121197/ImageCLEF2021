from sklearn.pipeline import Pipeline
from cnn_modality_clf import model

from cnn_modality_clf.config import config
from cnn_modality_clf.preprocessing import preprocessors as pp


pipe = Pipeline([
                ('dataset', pp.CreateDataset(config.IMAGE_SIZE)),
                ('cnn_model', model.cnn_clf)])