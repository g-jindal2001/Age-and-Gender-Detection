from datapreprocessing import *
from train_test_split import *
from model import *
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv2D, Activation, Dense, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.python.keras.models import Sequential

facial_age_path = r'C:\Users\gjadd\Downloads\facial-age'
utk_face_path = r'C:\Users\gjadd\Downloads\UTKFace'
image_dir_path = r"C:\Users\gjadd\Downloads\combined_faces"

facial_age_path_aws = r'/home/ubuntu/age_and_gender_detection/data/facial-age'
utk_face_path_aws = r'/home/ubuntu/age_and_gender_detection/data/UTKFace'
image_dir_path_aws = r'/home/ubuntu/age_and_gender_detection/data/combined_faces'

data_preprocess = DataPreProcessing(facial_age_path = facial_age_path, utk_face_path = utk_face_path)
data_preprocess.unique_ages_combined()
data_preprocess.combine_images()
data_preprocess.no_of_images_for_every_age()
data_preprocess.split_classes(n_classes=7)
#data_preprocess.merge_images(new_path=r"C:\Users\gjadd\Desktop\combined_images")

train_test_split = TrainTestSplit(image_dir_path = image_dir_path) #in production replace this with new_path of merge_images function
train_test_split.generate_df()
train_dataset, test_dataset = train_test_split.train_test_split(batch_size = 8)
#print(train_dataset) #shape=(None, None, None, 1)

cnn1 = Sequential()

cnn1.add(Conv2D(filters=8, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))    # 3rd dim = 1 for grayscale images.
cnn1.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
cnn1.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn1.add(MaxPool2D(pool_size=(2,2)))

cnn1.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn1.add(MaxPool2D(pool_size=(2,2)))

cnn1.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn1.add(MaxPool2D(pool_size=(2,2)))

cnn1.add(Flatten())    # GlobalAveragePooling2D(), compared to Flatten(), gave better accuracy values, and significantly reduced over-fitting and the no. of parameters.
cnn1.add(Dense(20, activation='relu'))
cnn1.add(Dense(7, activation='softmax')) #since we have 7 different class ranges

#classifier = AgeDetector()

cnn1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(cnn1.summary())

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
classifier_history = cnn1.fit(train_dataset,
                        batch_size=8,
                        validation_data=test_dataset,
                        epochs=30,
                        callbacks=[early_stop],
                        shuffle=False    # shuffle=False to reduce randomness and increase reproducibility
                       )