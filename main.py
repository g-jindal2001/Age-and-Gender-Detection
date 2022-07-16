from datapreprocessing import *
from datagen import *
from model import *

from tensorflow.python.keras.callbacks import EarlyStopping

facial_age_path = r'C:\Users\gjadd\Downloads\facial-age'
utk_face_path = r'C:\Users\gjadd\Downloads\UTKFace'
image_dir_path = r"C:\Users\gjadd\Downloads\combined_faces"

facial_age_path_aws = r'/home/ubuntu/age_and_gender_detection/data/facial-age'
utk_face_path_aws = r'/home/ubuntu/age_and_gender_detection/data/UTKFace'
image_dir_path_aws = r'/home/ubuntu/age_and_gender_detection/data/combined_faces'

data_preprocess = DataPreProcessing(utk_face_path = utk_face_path)
data_preprocess.count_of_img_for_every_age()
data_preprocess.split_classes(n_classes=7)
master_df = data_preprocess.generate_df()
print(master_df.head())

data_gen = DataGen(df = master_df) 
train_idx, valid_idx = data_gen.generate_split_indexes()
print(train_idx)
print(valid_idx)

classifier = AgeDetector().assemble_model(input_shape=(200, 200, 1))

classifier.compile(loss={
                        'age_branch_1': 'categorical_crossentropy',
                        'age_branch_2': 'categorical_crossentropy',
                    }, 
                    optimizer='adam',
                    loss_weights={
                        'age_branch_1': 2, 
                        'age_branch_2': 1,
                    },
                    metrics={
                        'age_branch_1': 'accuracy',
                        'age_branch_2': 'accuracy',
                    })

# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

train_dataset = data_gen.generate_images(image_idx = train_idx, is_training = True, batch_size = 8)
valid_dataset = data_gen.generate_images(image_idx = valid_idx, is_training = True, batch_size = 8)

classifier_history = classifier.fit(train_dataset,
                        steps_per_epoch=len(train_idx)//8,
                        validation_data=valid_dataset,
                        validation_steps=len(valid_idx)//8,
                        epochs=10,
                        #callbacks=[early_stop],
                        shuffle=False    # shuffle=False to reduce randomness and increase reproducibility
                       )