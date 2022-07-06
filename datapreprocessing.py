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
        self.facial_age_folder = None
        self.utk_face_folder = None
        self.facial_age_images = {}
        self.utk_face_images = {}
        self.combined_images = {}
        self.unique_ages = set()
        self.images_df = None
        self.classes_df = None
        self.new_path = None

    def unique_ages_combined(self):
        self.facial_age_folder = os.listdir(self.facial_age_path)
        self.utk_face_folder = os.listdir(self.utk_face_path)

        utkface_age_labels = np.array([])

        for age in self.facial_age_folder:
            temp_path = os.path.join(self.facial_age_path, age)
            n_images = len(os.listdir(temp_path))
            self.facial_age_images[int(age)] = n_images

        for image in self.utk_face_folder:
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

    def no_of_images_for_every_age(self):
        self.images_df = pd.DataFrame(self.combined_images.values(), index=self.combined_images.keys(), columns=['combined_images'])
        self.images_df['facial_age_images'] = pd.Series(self.facial_age_images)
        self.images_df['utkface_images'] = pd.Series(self.utk_face_images)

        self.images_df.fillna(0, inplace=True)
        self.images_df = self.images_df.astype(int)

    def generate_barplot_of_no_of_images_by_age(self):
        plt.figure(figsize=(30, 10))
        ax = sns.barplot(x=self.images_df.index, y=self.images_df['combined_images'], color='royalblue')

        ax.tick_params(axis='both', labelsize=12)
        ax.tick_params(axis='x', labelrotation=45)

        plt.xlabel("Age", fontsize=16)
        plt.ylabel("No. of Images", fontsize=16)

        plt.title("Barplot showing No. of images in Combined datasets (facial-age & UTKFace datasets) by Age", fontsize=18)
        plt.show()

    def split_classes(self, n_classes):
        series = self.images_df['combined_images']
        n_images = int(sum(series) / n_classes)

        print(f"Total no. of images in dataset\t= {sum(series)}")
        print(f"No. of classes desired\t\t= {n_classes}")
        print(f"Target no. of images per class\t>= {sum(series)}/{n_classes} = ~{n_images}")
        print()

        self.classes_df = pd.DataFrame(columns=['Age-ranges (classes)', 'No. of images', 'Class balance (%)'])
        age_index = 0

        for i in range(n_classes):

            # Storing the starting age of the class in a variable age_start.
            # Storing the current age being iterated in a variable age_current.
            # Keeping track of age_index variable so as not to let it go out of index.
            if age_index<=103:
                age_start = series.index[age_index] #ser.index will convert it into range of indexes and to get the index value we use ser.index[index_value]
                #print("Age Start", age_start)
                age_current = series.index[age_index]
                #print("Age Current", age_current)
            else:
                break

            # Initiating a new variable to keep track of no. of images added to current class.
            class_images = 0
            
            # Iterating through the ages in the given input series and adding up the no. of images
            # until it exceeds the target number of images per class, using the age_index and age_current variables.
            while class_images < n_images:
                class_images += series[age_current]
                #print("classes images", class_images)
                age_index += 1

                # Keeping track of age_index variable so as not to let it go out of index.
                if age_index<=103:
                    age_current = series.index[age_index]
                else:
                    break

            # Storing the ending age of the class in a variable age_end.
            # Keeping track of age_index variable so as not to let it go out of index.
            if age_index<=104:
                age_end = series.index[age_index-1]
                #print("Age End", age_end)
            else:
                break

            # Adding the above calculated variables into the dataframe for easier printing and analysis.
            self.classes_df.loc[i, 'Age-ranges (classes)'] = str(age_start)+" - "+str(age_end)
            self.classes_df.loc[i, 'No. of images'] = class_images
            self.classes_df.loc[i, 'Class balance (%)'] = round((class_images / sum(series)) * 100, 2)

        return self.classes_df

    def merge_images(self, new_path):
        self.new_path = new_path
        isExist = os.path.exists(self.new_path)
        if not isExist:
            os.makedirs(self.new_path)

        age_file_counter = [1] * 117

        print("\nMerging images from facial-age dataset into directory.\n")

        for age in self.facial_age_folder:
            age_path = os.path.join(self.facial_age_path, age)

            img_files = os.listdir(age_path)

            for img in img_files:

                img_src = os.path.join(age_path, img)

                new_filename = str(int(age)) + "_" + str(age_file_counter[int(age)]) + ".jpg"
                age_file_counter[int(age)] += 1

                img_dest = os.path.join(self.new_path, new_filename)

                # Converting the .PNG images to .JPG so as to maintain consistency with of filetype throughout the combined datasets.
                png_image = cv2.imread(img_src)
                cv2.imwrite(img_dest, png_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print("Merging images from UTKFace dataset into directory.\n")

        for img in self.utk_face_folder:

            file_type = img.split(".")[-1]
            age = img.split("_")[0]

            img_src = os.path.join(self.utk_face_path, img)

            new_filename = age + "_" + str(age_file_counter[int(age)]) + "." + file_type
            age_file_counter[int(age)] += 1

            img_dest = os.path.join(self.new_path, new_filename)

            shutil.copy(img_src, img_dest)

        print("Done merging images from both datasets into directory.")

    def get_all_file_paths(self):
        file_paths = []

        for root, dirs, files in os.walk(r"C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\combined_faces\content\combined_faces"):# replace with self.new_path while use in production
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)

        return file_paths

    def display(self):
        print(sum(self.combined_images.values()))
        print(self.images_df.shape)
        print(self.classes_df.shape)

