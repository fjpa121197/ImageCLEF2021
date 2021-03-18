import joblib

from models import densenet_121
from config import config_default_cnn, config_densenet_121
from preprocessing import data_management as dm 
from preprocessing import preprocessors as pp
import sys


def run_training(model: str, save_result: bool = True):

    train_generator = dm.image_generator(data_path = config_densenet_121.DATA_FOLDER, 
                                        image_size = config_densenet_121.IMAGE_SIZE,
                                        batch_size = config_densenet_121.BATCH_SIZE, subset = 'training')
    validation_generator = dm.image_generator(data_path = config_densenet_121.DATA_FOLDER, 
                                            image_size = config_densenet_121.IMAGE_SIZE,
                                            batch_size = config_densenet_121.BATCH_SIZE, subset = 'validation')

    if model == 'densenet121':
        densenet_121.cnn_clf.fit(train_generator,validation_data = validation_generator)
    elif model == 'default':
        pass
    else:
        pass

    if save_result:
        joblib.dump(enc, config.ENCODER_PATH)
        dm.save_pipeline_keras(pipe.pipe)


if __name__ == '__main__':
    model = sys.argv[1]
    run_training(model = model)