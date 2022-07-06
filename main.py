from datapreprocessing import *
from train_test_split import *

data_preprocess = DataPreProcessing(facial_age_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\facial-age\facial-age', utk_face_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\UTKFace\UTKFace')
data_preprocess.unique_ages_combined()
data_preprocess.combine_images()
data_preprocess.no_of_images_for_every_age()
data_preprocess.split_classes(n_classes=7)

#data_preprocess.merge_images(new_path=r"C:\Users\gjadd\Desktop\combined_images")

train_test_split = TrainTestSplit(image_dir_path = r"C:\Users\gjadd\Downloads\combined_faces")
train_test_split.generate_df()
train_dataset, test_dataset = train_test_split.train_test_split()

