from tensorflow.keras import models, layers, optimizers, callbacks, wrappers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121


class DenseNetClassifier():
    def __init__(self):
        self._image_size = 256
        self._weigths = 'imagenet'
        self._loss = 'categorical_crossentropy'
        self._optimizer = optimizers.Adam(learning_rate=0.001)
        self._metrics = ['acc']
        self._model = None
        self._callbacks = [callbacks.EarlyStopping(monitor = 'val_loss', patience = 2, restore_best_weights = True)]

    def build_model(self):
        #Download pretrained DenseNet 121 model with imagenet weights (default) and image size
        base_model = DenseNet121(include_top = False, weights = self._weigths, 
                                input_shape= (self._image_size,self._image_size,3))
        model = layers.GlobalAveragePooling2D()(base_model.output)
        model = layers.Dense(64, activation= 'relu')(model)
        model = layers.Dropout(0.5, seed = 14)(model)
        prediction_layer = layers.Dense(7, activation= 'softmax')(model)
        self._model = models.Model(inputs = base_model.input, outputs = prediction_layer)
        #print(self._model.summary())
        self._model.compile(loss = self._loss, optimizer = self._optimizer, metrics = self._metrics)

        return self._model

    def train(self, train_generator, validation_generator):

        history = self._model.fit(train_generator, train_generator_1, steps_per_epoch=10, epochs=10,
                                validation_data=validation_generator, callbacks = self._callbacks,
                                validation_steps=2, verbose = 1)

        return history


        