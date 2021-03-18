import os
from numpy.random import seed
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout

seed(1)
tf.random.set_seed(14)

train_datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255)

train_generator_1 = ImageDataGenerator(validation_split=0.2,rescale=1./255).flow_from_directory(directory = '../dataset/ImageCLEF2020_Train_Images/Train/',
                                                    target_size=(256, 256), 
                                                    batch_size=10,
                                                    subset = 'training', seed = 14,
                                                    class_mode='categorical')

validation_generator_1 = ImageDataGenerator(validation_split=0.2,rescale=1./255).flow_from_directory(directory='../dataset/ImageCLEF2020_Train_Images/Train/',
                                                        target_size=(256, 256),
                                                        batch_size=10,
                                                        subset = 'validation', seed = 14,
                                                        class_mode='categorical')

test_generator_4 = ImageDataGenerator(rescale=1./255).flow_from_directory(directory = '../dataset/ImageCLEF2020_Test_Images/',
                                                                        target_size = (256,256), shuffle = False, class_mode = 'categorical', batch_size = 32)



model_base_densenet = DenseNet121(include_top = False, weights = '../weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape= (256,256,3))

model_densenet121 = GlobalAveragePooling2D()(model_base_densenet.output)
model_densenet121 = Dense(64, activation= 'relu')(model_densenet121)
model_densenet121 = Dropout(0.5)(model_densenet121)
predictions = Dense(7, activation= 'softmax')(model_densenet121)
model_3 = models.Model(inputs=model_base_densenet.input, outputs=predictions)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.90,
    staircase=True)


model_3.compile( loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(learning_rate=lr_schedule),
              metrics=['acc'])
#Early stopping based on val_loss, however, this metrics and params might change. Also, reducing learning rate might be implemented
callback_earlystop = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights= True, verbose=1)




history_3 = model_3.fit(train_generator_1,
              steps_per_epoch=100,
              epochs=100,
              validation_data=validation_generator_1, callbacks = [callback_earlystop],
              validation_steps=10)

model_3.save('ImageCLEF_Modality_Detection.h5')

acc = history_3.history['acc']
val_acc = history_3.history['val_acc']
loss = history_3.history['loss']
val_loss = history_3.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('this is a test2.png', bbox_inches='tight')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('this is a test1.png', bbox_inches='tight')
plt.show()


test_steps_per_epoch = np.math.ceil(test_generator_4.samples / test_generator_4.batch_size) # You can either supply each image and make a single prediction or make a batch prediction

predictions = model_3.predict(test_generator_4,steps = test_steps_per_epoch)
prediction_classes = np.argmax(predictions, axis = 1) # Returns true for the index with the highest probability

true_classes_4 = test_generator_4.classes # Gets the available classes passed to the generator
class_labels_4 = list(test_generator_4.class_indices.keys()) # Gets the label ffor each class
report_4 = metrics.classification_report(true_classes_4,prediction_classes, target_names = class_labels_4)
print("Based model classification report")
print(report_4)