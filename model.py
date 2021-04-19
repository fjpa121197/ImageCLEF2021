from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
import numpy as np
import pandas as pd
import csv
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import tensorflow
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam

from multioutput import ConceptDetGenerator



class MultiOutputModel():

    def __init__(self, gen, semanticTypes):
        self.generator = gen
        self.semanticTypes = semanticTypes

    def post_densenet_layer(self, x):

        x = GlobalAveragePooling2D()(x)
        
        return x

    def branch(self, inputs, num_classes, name):


        x = Dense(512, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(num_classes, activation='sigmoid', name = name)(x)

        return x

    def full_model(self, width, height, num_of_branches):



        base_model = DenseNet121(include_top=False, weights='imagenet',input_shape=(width,height,3)) # Load DenseNet-121 model with same input shape as stated above
        base_model.trainable = False

        feature_map = self.post_densenet_layer(base_model.output)

        branches = []
        for branch in self.semanticTypes:
            branches.append(self.branch(feature_map, len(self.generator.mlb_dic[branch].classes_), self.generator.st_dic[branch] + "_output"))
 
        model = Model(inputs = base_model.input, outputs = branches, name = "multioutput_model")

        return model


batch_size = 32
valid_batch_size = 32

semanticTypes = ["Diagnostic Procedure","Body Part, Organ, or Organ Component","ee"]


cg = ConceptDetGenerator(semanticTypes)
cg.create_dict_of_concepts()
cg.generate_split_idx()
cg.generate_mlb()

with open("mlb_dic_full.pkl", 'wb') as f:
    pickle.dump(cg.mlb_dic, f)

train_gen = cg.generate_images("train", batch_size = batch_size)
valid_gen = cg.generate_images("val", batch_size = valid_batch_size)


model = MultiOutputModel(cg, semanticTypes).full_model(64,64,2)

tf.keras.utils.plot_model(model, to_file="./model_3b.png", show_shapes=True)

#print(model.summary())


init_lr = 1e-4
epochs = 100
opt = Adam(lr=init_lr, decay=init_lr / epochs)

model.compile(optimizer=opt, 
              loss={
                  'dp_output': 'binary_crossentropy', 
                  'bpo_output': 'binary_crossentropy',
                  'ee_output': 'binary_crossentropy'},
              loss_weights={
                  'dp_output': 1,
                  'bpo_output': 1, 
                  'ee_output': 1},
              metrics={
                  'dp_output': 'accuracy',
                  'bpo_output': 'accuracy', 
                  'ee_output': 'accuracy'})

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]

history = model.fit(train_gen,
                    steps_per_epoch=len(cg.img_array_train)//batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=valid_gen,             
                    validation_steps=len(cg.img_array_val)//valid_batch_size)