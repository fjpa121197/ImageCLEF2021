from sklearn.pipeline import Pipeline
from cnn_modality_clf import model





pipe = Pipeline([
                ('dataset', pp.CreateDataset(config.IMAGE_SIZE)),
                ('cnn_model', model.cnn_clf)])