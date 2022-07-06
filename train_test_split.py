import os
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

import tensorflow as tf

def class_labels(age):
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

def _parse_function(filename, label):#convert file names to actual tensors(numpy arrays)
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=1)    # channels=1 to convert to grayscale, channels=3 to convert to RGB.
    # image_resized = tf.image.resize(image_decoded, [200, 200])
    label = tf.one_hot(label, 7)

    return image_decoded, label

class TrainTestSplit:
    def __init__(self, image_dir_path):
        self.image_dir_path = image_dir_path
        self.image_names = None
        self.master_df = None

    def generate_df(self):
        self.image_names = os.listdir(self.image_dir_path)
        master_df = pd.DataFrame()
        master_df['filename'] = self.image_names
        master_df['age'] = master_df['filename'].map(lambda img_name : np.uint8(img_name.split("_")[0]))
        master_df['target'] = master_df['age'].map(class_labels)
        master_df['filename'] = master_df['filename'].map(lambda img_name: os.path.join(self.image_dir_path, img_name))

        master_df = shuffle(master_df, random_state=42).reset_index(drop=True)
        #print(master_df.head())
        self.master_df = master_df

    def train_test_split(self, test_size=0.3):
        X = self.master_df[['filename', 'age']]
        y = self.master_df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        train_filenames_tensor = tf.constant(list(X_train['filename']))
        train_labels_tensor = tf.constant(list(y_train))
        test_filenames_tensor = tf.constant(list(X_test['filename']))
        test_labels_tensor = tf.constant(list(y_test))

        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames_tensor, train_labels_tensor))
        train_dataset = train_dataset.map(_parse_function)
        train_dataset = train_dataset.repeat(3)
        train_dataset = train_dataset.batch(512)

        test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames_tensor, test_labels_tensor))
        test_dataset = test_dataset.map(_parse_function)
        test_dataset = test_dataset.repeat(3)
        test_dataset = test_dataset.batch(512)

        return (train_dataset, test_dataset)

        #print([x.get_shape().as_list() for x in train_dataset._tensors])

        #x = np.array([np.array(Image.open(fname)) for fname in list(X_train['filename'])]) #returns with shape of (23440, 200, 200, 3)

    


    
