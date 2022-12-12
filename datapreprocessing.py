import numpy as np
import os
import pandas as pd

np.random.seed(42)  # random_seed is used to ensure that when we re-run this model we will get same values instead of different values, 42 is just the most common value for random_seed


class DataPreProcessing:
    def __init__(self, utk_face_path):
        self.utk_face_path = utk_face_path
        self.utk_face_images = None
        self.images_df = None

    def age_gender_race_split(self, image_name):
        image_labels = image_name.split('_')
        age = image_labels[0]
        gender = image_labels[1]

        return (age, gender)

    def count_of_img_for_every_age(self):
        self.utk_face_images = os.listdir(self.utk_face_path)

        utkface_age_labels = np.array([])
        utk_face_arr = {}

        for image in self.utk_face_images:
            age, _ = self.age_gender_race_split(image)
            utkface_age_labels = np.append(utkface_age_labels, age)

        utkface_ages_counts = pd.Series(utkface_age_labels).value_counts()

        for age, counts in utkface_ages_counts.items():
            utk_face_arr[int(age)] = counts

        utk_face_arr = dict(sorted(utk_face_arr.items()))

        self.images_df = pd.DataFrame(utk_face_arr.values(
        ), index=utk_face_arr.keys(), columns=['combined_images'])

    def split_classes(self, n_classes):  # used for defining class_labels in datagen.py
        series = self.images_df['combined_images']
        n_images = int(sum(series) / n_classes)

        print(f"Total no. of images in dataset\t= {sum(series)}")
        print(f"No. of classes desired\t\t= {n_classes}")
        print(
            f"Target no. of images per class\t>= {sum(series)}/{n_classes} = ~{n_images}")
        print()

        classes_df = pd.DataFrame(
            columns=['Age-ranges (classes)', 'No. of images', 'Class balance (%)'])
        age_index = 0

        for i in range(n_classes):

            # Storing the starting age of the class in a variable age_start.
            # Storing the current age being iterated in a variable age_current.
            # Keeping track of age_index variable so as not to let it go out of index.
            if age_index <= 103:
                # ser.index will convert it into range of indexes and to get the index value we use ser.index[index_value]
                age_start = series.index[age_index]
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
                if age_index <= 103:
                    age_current = series.index[age_index]
                else:
                    break

            # Storing the ending age of the class in a variable age_end.
            # Keeping track of age_index variable so as not to let it go out of index.
            if age_index <= 104:
                age_end = series.index[age_index-1]
                #print("Age End", age_end)
            else:
                break

            # Adding the above calculated variables into the dataframe for easier printing and analysis.
            classes_df.loc[i,
                           'Age-ranges (classes)'] = str(age_start)+" - "+str(age_end)
            classes_df.loc[i, 'No. of images'] = class_images
            classes_df.loc[i, 'Class balance (%)'] = round(
                (class_images / sum(series)) * 100, 2)

        print(classes_df)

    def class_labels(self, age):
        if 1 <= age <= 2:
            return 0
        elif 3 <= age <= 9:
            return 1
        elif 10 <= age <= 20:
            return 2
        elif 21 <= age <= 27:
            return 3
        elif 28 <= age <= 45:
            return 4
        elif 46 <= age <= 65:
            return 5
        else:
            return 6

    def generate_df(self):

        master_df = pd.DataFrame()

        master_df['filename'] = self.utk_face_images
        master_df['age'] = master_df['filename'].map(
            lambda img_name: np.uint8(img_name.split("_")[0]))
        master_df['gender'] = master_df['filename'].map(
            lambda img_name: np.uint8(img_name.split("_")[1]))
        master_df['target'] = master_df['age'].map(self.class_labels)
        master_df['filename'] = master_df['filename'].map(
            lambda img_name: os.path.join(self.utk_face_path, img_name))

        #master_df = shuffle(master_df, random_state=42).reset_index(drop=True)

        return master_df
