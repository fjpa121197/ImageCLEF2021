from tensorflow.keras import models, layers, optimizers, callbacks, wrappers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121


class DenseNetClassifier():
    def __init__(self, train_generator, validation_generator):
        self._train_generator = train_generator
        self._validation_generator = validation_generator
        self._image_size = 256
        self._weigths = 'imagenet'
        self._loss = 'categorical_crossentropy'
        self._optimizer = optimizers.Adam(learning_rate=0.001)
        self._metrics = ['acc']
        self._model = None
        self._train_generator = None
        self._validation_generator = None
        self._callbacks = [callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)]
        self._basemodel = DenseNet121(include_top = False, weights = self._weigths, 
                                input_shape= (self._image_size,self._image_size,3))

    def build_classifier(self):

        #Build
        #Download pretrained DenseNet 121 model with imagenet weights (default) and image size
        model = layers.GlobalAveragePooling2D()(self._basemodel.output)
        model = layers.Dense(64, activation= 'relu')(model)
        model = layers.Dropout(0.5, seed = 14)(model)
        prediction_layer = layers.Dense(7, activation= 'softmax')(model)
        self._model = models.Model(inputs = self._basemodel.input, outputs = prediction_layer)
        self._model.compile(loss = self._loss, optimizer = self._optimizer, metrics = self._metrics)

        return self

        """
        #Train section
        self._model.fit(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator, callbacks = self._callbacks,
                              validation_steps=10)

        """
    def train(self):
        #Train section
        self._model.fit(self._train_generator,
                              steps_per_epoch=5,
                              epochs=5,
                              validation_data=self._validation_generator, callbacks = self._callbacks,
                              validation_steps=2)

        return self._model
