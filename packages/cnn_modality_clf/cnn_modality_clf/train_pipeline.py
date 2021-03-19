from models.densenet_121 import DenseNetClassifier
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


    if model_arg == 'densenet121':
        base_model = DenseNetClassifier()
        clf = base_model.build_classifier()
        trained_clf = clf.train(train_generator, validation_generator)
        trained_clf.save('This-is-model.h5')
         
if __name__ == '__main__':
    model_arg = sys.argv[1]
    run_training(model_arg = model_arg)