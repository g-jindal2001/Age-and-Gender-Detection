from datapreprocessing import *
from train_test_split import *

data_preprocess = DataPreProcessing(facial_age_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\facial-age\facial-age', utk_face_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\UTKFace\UTKFace')
data_preprocess.unique_ages_combined()
data_preprocess.combine_images()
data_preprocess.no_of_images_for_every_age()
combined_classes = data_preprocess.split_classes(n_classes=12)
print(combined_classes)

data_preprocess.display()
#data_preprocess.merge_images(new_path=r"C:\Users\gjadd\Desktop\combined_images")
file_paths = data_preprocess.get_all_file_paths() #always use this function after merge images in production

train_test_split = TrainTestSplit(file_paths = file_paths)

