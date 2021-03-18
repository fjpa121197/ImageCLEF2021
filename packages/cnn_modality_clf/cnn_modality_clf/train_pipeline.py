from models import densenet_121
from config import config_default_cnn, config_densenet_121
from preprocessing import data_management as dm 
import sys


def run_training(model: str, save_result: bool = True):

    train_generator = dm.image_generator(data_path = config_densenet_121.DATA_FOLDER, 
                                        image_size = config_densenet_121.IMAGE_SIZE,
                                        batch_size = config_densenet_121.BATCH_SIZE, subset = 'training')
    
    validation_generator = dm.image_generator(data_path = config_densenet_121.DATA_FOLDER, 
                                            image_size = config_densenet_121.IMAGE_SIZE,
                                            batch_size = config_densenet_121.BATCH_SIZE, subset = 'validation')

    print(densenet_121.cnn_clf.summary())
    if model == 'densenet121':
        densenet_121.cnn_clf.fit(train_generator,validation_data = validation_generator)
    elif model == 'default':
        pass
    else:
        pass


if __name__ == '__main__':
    model = sys.argv[1]
    print(model)
    run_training(model = model)