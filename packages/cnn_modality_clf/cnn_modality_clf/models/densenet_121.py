from tensorflow.keras import models, layers, optimizers, callbacks, wrappers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121

from config import config_densenet_121

def densenet_121_model(image_size = 256, weights = 'imagenet'):
    
    #Download pretrained DenseNet 121 model with imagenet weights (default) and image size
    base_model = DenseNet121(include_top = False, weights = weights, input_shape= (image_size,image_size,3))
    model = layers.GlobalAveragePooling2D()(base_model.output)
    model = layers.Dense(64, activation= 'relu')(model)
    model = layers.Dropout(0.5, seed = 14)(model)
    prediction_layer = layers.Dense(7, activation= 'softmax')(model)
    model = models.Model(inputs = base_model.input, outputs = prediction_layer)

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(learning_rate = 0.001),
                metrics = ['acc'])

    

    return model


earlystop = callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights= True)
callbacks = [earlystop]

cnn_clf = KerasClassifier(build_fn=densenet_121_model,
                          validation_steps=10,
                          epochs=config_densenet_121.EPOCHS,
                          steps_per_epoch = config_densenet_121.EPOCHS,
                          verbose=1,  # progress bar - required for CI job
                          callbacks = [callbacks]
                          )

if __name__ == '__main__':
    model = densenet_121_model()
    model.summary()
