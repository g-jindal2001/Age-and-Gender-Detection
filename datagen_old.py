import numpy as np
from PIL import Image

from tensorflow.python.keras.utils.np_utils import to_categorical

class DataGen:
    def __init__(self, df):
        self.master_df = df

    def generate_split_indexes(self):
        p = np.random.permutation(len(self.master_df))
        train_up_to = int(len(self.master_df) * 0.7)
        train_idx, valid_idx  = p[:train_up_to], p[train_up_to:]
        return train_idx, valid_idx

    def preprocess_image(self, img_path):
        im = Image.open(img_path).convert('L')
        im = np.array(im) / 255.0
        return im

    def generate_images(self, image_idx, is_training, batch_size=512):
        images, ages, genders = [], [], []
        while True:
            for idx in image_idx:
                person = self.master_df.iloc[idx]
                age = person['target']
                gender = person['gender']
                file = person['filename']

                im = self.preprocess_image(file)

                ages.append(to_categorical(age, 7))
                genders.append(to_categorical(gender, 2))
                images.append(im)
                
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(genders)]
                    images, ages, genders = [], [], []

            if not is_training:
                break

    


    
