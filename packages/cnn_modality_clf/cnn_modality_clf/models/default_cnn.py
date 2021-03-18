from tensorflow.keras import models, layers, optimizers, callbacks, wrappers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from cnn_modality_clf.config import config


# TODO: [CLEF21-2] Check how to set a seed to repoduce results


def cnn_model(kernel_size=(3, 3), pool_size=(2, 2), first_filters=32, second_filters=64,
              dropout_conv=0.3, dropout_dense=0.3, image_size=64):
    
    model = models.Sequential()
    model.add(layers.Conv2D(first_filters, kernel_size, activation= 'relu', 
                            input_shape = (image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    model.add(layers.Conv2D(second_filters, kernel_size, activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    model.add(layers.Conv2D(second_filters, kernel_size, activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(second_filters, activation = 'relu'))
    model.add(layers.Dense(7, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['accuracy'])

    return model


cnn_clf = KerasClassifier(build_fn=cnn_model,
                          batch_size=config.BATCH_SIZE,
                          validation_split=10,
                          epochs=config.EPOCHS,
                          verbose=1,  # progress bar - required for CI job
                          image_size=config.IMAGE_SIZE
                          )

if __name__ == '__main__':
    model = cnn_model()
    model.summary()
