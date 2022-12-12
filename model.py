from tensorflow.python.keras.layers import Conv2D, Activation, Dense, MaxPool2D, Flatten, Input, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model


class AgeDetector():
    def hidden_layers_age(self, inputs_age):
        x = Conv2D(32, (3, 3))(inputs_age)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3))(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3))(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        return x

    def hidden_layers_gender(self, inputs_gender):
        y = Conv2D(32, (3, 3))(inputs_gender)
        y = Activation("relu")(y)
        y = MaxPool2D(pool_size=(2, 2))(y)

        y = Conv2D(64, (3, 3))(y)
        y = Activation("relu")(y)
        y = MaxPool2D(pool_size=(2, 2))(y)

        y = Conv2D(128, (3, 3))(y)
        y = Activation("relu")(y)
        y = MaxPool2D(pool_size=(2, 2))(y)

        y = Conv2D(256, (3, 3))(y)
        y = Activation("relu")(y)
        y = MaxPool2D(pool_size=(2, 2))(y)

        y = Flatten()(y)

        return y

    def assemble_model(self, input_shape, no_of_classes):

        inputs = Input(shape=input_shape, name="img_input")

        x = self.hidden_layers_age(inputs)
        y = self.hidden_layers_gender(inputs)

        age_dense = Dense(128, activation='relu')(x)  #age
        age_output = Dense(no_of_classes, activation='softmax',
                           name="age_output")(age_dense)  #age

        gender_dense = Dense(128, activation='relu')(y)  #gender
        gender_output = Dense(2, activation='sigmoid',
                              name="gender_output")(gender_dense)  #gender

        model = Model(inputs=inputs, outputs=[
                      age_output, gender_output], name="age_gender_model")
        # print(model.summary())

        return model
