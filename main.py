from datapreprocessing import *

data_preprocess = DataPreProcessing(facial_age_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\facial-age\facial-age', utk_face_path = r'C:\Users\gjadd\Downloads\ZIPPED_DATASETS-20220704T123949Z-002\ZIPPED_DATASETS\UTKFace\UTKFace')
data_preprocess.unique_ages_combined()
data_preprocess.combine_images()
data_preprocess.generate_df()
combined_classes = data_preprocess.split_classes(n_classes=12)
print(combined_classes)

data_preprocess.display()
#data_preprocess.merge_images(new_path=r"C:\Users\gjadd\Desktop\combined_images")
