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

#Declare paths

training_concepts_path = './ImageCLEF2021_ConceptDetection_Training-Set/Training_Set_Concepts.csv'
training_images_path = './ImageCLEF2021_ConceptDetection_Training-Set/Training-Images/'

validation_concepts_path = './ImageCLEF2021_ConceptDetection_Validation-Set/Validation_Set_Concepts.csv'
validation_images_path = './ImageCLEF2021_ConceptDetection_Validation-Set/Validation-Images/'

df = pd.read_csv("./Training_Set_Concepts_f.csv", header = 0)
#semanticTypes = ["Body Location or Region", "Finding", "Body Part, Organ, or Organ Component","Diagnostic Procedure", "Disease or Syndrome", "ee"]



class ConceptDetGenerator():
    def __init__(self, semanticTypes):

        """
        image_id_concepts_dict:     Dictonary to obtain validation image info from image name
        image_id_concepts_val_dict: Dictonary to obtain train image info from image name
        mlb_dic:                    Dictonary containing MultiLabelBinarizer for each semanticType
        img_array_train:            Array of name of files for training.
        img_array_val:              Array of name of files for validation.
        img_array_test:             Array of name of files for training.

        df:                         DataFrame previosly created to obtain all existing classes and making the MultiLabelBinarizer 
        semanticTypes:              SemanticTypes in which the data will be group on
        """
        
        self.image_id_concepts_dict = dict()
        self.image_id_concepts_val_dict = dict()
        self.mlb_dic = {}
        self.img_array_train = os.listdir(training_images_path)
        self.img_array_val = os.listdir(validation_images_path)
        self.img_array_test = []
        self.df = pd.read_csv("./Training_Set_Concepts_f.csv", header = 0)
        self.semanticTypes = semanticTypes
        self.st_dic = {'Body Location or Region': 'blr', 'Finding': 'f', 'Body Part, Organ, or Organ Component': 'bpo', "Diagnostic Procedure" : "dp", "Disease or Syndrome" : "ds",  "ee": "ee" }




    def create_dict_of_concepts(self):

        """
        Creates dictionaries with the name of the image and its concepts. 
        """
        with open(training_concepts_path, "r", encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                self.image_id_concepts_dict[line[0]+'.jpg'] = list(line[1].split(';'))

        with open(validation_concepts_path, "r", encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                self.image_id_concepts_val_dict[line[0]+'.jpg'] = list(line[1].split(';'))


    def preprocess_image(self, image):
        """
        Process image for DenseNet
        """
        img = tensorflow.keras.preprocessing.image.load_img(path = image, target_size = (64,64)) # Load actual image
        img = tensorflow.keras.preprocessing.image.img_to_array(img) # Transform image to array of shape (input_shape)
        img = tensorflow.keras.applications.densenet.preprocess_input(img) # This preprocess_input normalizes the pixel values based on imagenet dataset and rescale to a 0-1 values.
        return img


    def generate_split_idx(self):
        """
        Splits the validation dataset for testing
        """
        p = np.random.permutation(len(self.img_array_val))
        val_up_to = int(len(self.img_array_val) * 0.5)
        val_idx = p[:val_up_to]
        test_idx = p[val_up_to:]


        self.img_array_test = [self.img_array_val[i] for i in test_idx]
        self.img_array_val = [self.img_array_val[i] for i in val_idx]


    def generate_mlb(self):
        """
        Creates the mlb dictionary to create the MultiLabelBinarizer for each semanticType
        """
        for typec in self.semanticTypes:
            self.mlb_dic[typec] = MultiLabelBinarizer()

            if typec != "ee":
                self.mlb_dic[typec].fit_transform([list(df[df["SemanticType"] == typec]["Concept"].values)])
            else: 
                self.mlb_dic[typec].fit_transform([list(df[~df["SemanticType"].isin(self.semanticTypes)]["Concept"].dropna().values)])



    def generate_images(self, selection, is_training = True,batch_size=16):

        """
            blr: Body Location, or Region
            f: Finding
            bpo: Body Part, Organ or Organ Component
            dp: Diagnostic Procedure
            ds: Disease or Syndrome
            ee: Everthing else
        """


        dp_list = open("dp.txt", "r").read().split("\n")
        bpo_list = open("bpo.txt", "r").read().split("\n")

        images, blr, f, bpo, dp, ds, ee = [],[],[],[],[],[],[]
        curr_array = []
        features = []
        all_groups = []


        if selection == "train":
            curr_array = self.img_array_train
            curr_dic = self.image_id_concepts_dict
            path = training_images_path
        elif selection == "val":
            curr_array = self.img_array_val
            curr_dic = self.image_id_concepts_val_dict
            path = validation_images_path
        else:
            curr_array = self.img_array_test

        while True:
            for ins in curr_array:


                for i in self.semanticTypes:
                    if i != "ee":
                        locals()[self.st_dic[i] + "_and_list"] = np.array(list(set(curr_dic[ins]) & set(locals()[self.st_dic[i] + "_list"])))                    
                        all_groups = all_groups + list(locals()[self.st_dic[i] + "_and_list"])
                        locals()[self.st_dic[i]].append(self.mlb_dic[i].transform(np.reshape(locals()[self.st_dic[i] + "_and_list"], (1, locals()[self.st_dic[i] + "_and_list"].shape[0]))))

                ee1 = np.array(list(set(curr_dic[ins]) - set(all_groups)))
                ee.append(self.mlb_dic["ee"].transform(np.reshape(ee1, (1, ee1.shape[0]))))

                im = self.preprocess_image(path + ins)

                images.append(im)

                if len(images) >= batch_size:

                    for i in self.semanticTypes:
                        features.append(np.squeeze(np.array(locals()[self.st_dic[i]])))

                    yield np.array(images), features

                    features.clear()
                    all_groups.clear()

                    images, blr, f, bpo, dp, ds, ee = [],[],[],[],[],[],[]

            if not is_training:
                break


        

                 



