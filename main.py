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

data_preprocess = DataPreProcessing(utk_face_path=utk_face_path)
data_preprocess.count_of_img_for_every_age()
data_preprocess.split_classes(n_classes=7)
master_df = data_preprocess.generate_df()
print(master_df.head())

data_gen = DataGen(df=master_df)
#train_idx, valid_idx = data_gen.generate_split_indexes()

classifier = AgeDetector().assemble_model(input_shape=(200, 200, 1))

classifier.compile(loss={
    'age_output': 'categorical_crossentropy',
    'gender_output': 'binary_crossentropy',
},
    optimizer='adam',  # Different optimizers in multi-output model is not allowed
    # loss_weights={
    #     'age_output': 2,
    #     'gender_output': 1,
    # },
    metrics={
    'age_output': 'accuracy',
    'gender_output': 'accuracy',
})

# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

#train_dataset = data_gen.generate_images(image_idx = train_idx, is_training = True, batch_size = 8)
#valid_dataset = data_gen.generate_images(image_idx = valid_idx, is_training = True, batch_size = 8)

train_dataset, valid_dataset = data_gen.train_test_split(
    test_size=0.2, batch_size=8) #Change batch size according to your system

classifier_history = classifier.fit(train_dataset,
                                    # steps_per_epoch=len(train_idx)//8,
                                    validation_data=valid_dataset,
                                    # validation_steps=len(valid_idx)//8,
                                    epochs=10,
                                    # callbacks=[early_stop],
                                    # shuffle=False    # shuffle=False to reduce randomness and increase reproducibility
                                    )
