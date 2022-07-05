import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import seaborn as sns
import cv2
import shutil

np.random.seed(42) #random_seed is used to ensure that when we re-run this model we will get same values instead of different values, 42 is just the most common value for random_seed

def age_gender_race_split(image_name):
    image_labels = image_name.split('_')
    age = image_labels[0]
    gender = image_labels[1]
    race = image_labels[2]

    return (age, gender, race)

class DataPreProcessing:
    def __init__(self,facial_age_path, utk_face_path):
        self.facial_age_path = facial_age_path
        self.utk_face_path = utk_face_path
        self.facial_age_images = {}
        self.utk_face_images = {}
        self.combined_images = {}
        self.unique_ages = set()

    def unique_ages_combined(self):
        facial_age_folder = os.listdir(self.facial_age_path)
        utk_face_folder = os.listdir(self.utk_face_path)

        utkface_age_labels = np.array([])

        for age in facial_age_folder:
            temp_path = os.path.join(self.facial_age_path, age)
            n_images = len(os.listdir(temp_path))
            self.facial_age_images[int(age)] = n_images

        for image in utk_face_folder:
            age, _, __ = age_gender_race_split(image)
            utkface_age_labels = np.append(utkface_age_labels, age)

        utkface_ages_counts = pd.Series(utkface_age_labels).value_counts()

        for age, counts in utkface_ages_counts.items():
            self.utk_face_images[int(age)] = counts

        facial_age_ages = list(self.facial_age_images.keys())
        utk_face_ages = list(self.utk_face_images.keys())

        facial_age_ages.extend(utk_face_ages)

        self.unique_ages = set(facial_age_ages)

    def combine_images(self):
        for age in self.unique_ages:
            fc_image = 0
            utk_image = 0

            # Using try-except loop to avoid KeyError in case a particular age value does not exist in the dictionary.
            try:
                fc_image = self.facial_age_images[age]
            except:
                pass
            
            try:
                utk_image = self.utk_face_images[age]
            except:
                pass
            
            # Summing up the no. of images for the age label.
            self.combined_images[age] = fc_image + utk_image

    def display(self):
        print(sum(self.combined_images.values()))

data_preprocess = DataPreProcessing(facial_age_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\facial-age\facial-age', utk_face_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\UTKFace\UTKFace')
data_preprocess.unique_ages_combined()
data_preprocess.combine_images()
data_preprocess.display()