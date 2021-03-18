from models import densenet_121
from config import config_default_cnn, config_densenet_121
from preprocessing import data_management as dm 
import sys


def run_training(model_arg: str, save_result: bool = True):

    train_generator = dm.image_generator(data_path = config_densenet_121.DATA_FOLDER, 
                                        image_size = config_densenet_121.IMAGE_SIZE,
                                        batch_size = config_densenet_121.BATCH_SIZE, subset = 'training')
    
    validation_generator = dm.image_generator(data_path = config_densenet_121.DATA_FOLDER, 
                                            image_size = config_densenet_121.IMAGE_SIZE,
                                            batch_size = config_densenet_121.BATCH_SIZE, subset = 'validation')

    print(densenet_121.cnn_clf.summary())
    if model_arg == 'densenet121':
        model = densenet_121.DenseNetClassifier().build_model()
        print(model.summary())
    


if __name__ == '__main__':
    model_arg = sys.argv[1]
    print(model_arg)
    run_training(model_arg = model_arg)