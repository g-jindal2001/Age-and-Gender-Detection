from tensorflow.python.keras.layers import Conv2D, Activation, Dense, MaxPool2D, Flatten, Input, Dropout
from tensorflow.python.keras.models import Model

class AgeDetector():
    def hidden_layers(self,inputs):
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        return x

    def age_branch(self, inputs, n_classes=7):

        x = self.hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(20)(x)
        x = Activation("relu")(x)
        x = Dense(n_classes)(x)
        x = Activation("softmax", name="age_branch_1")(x)

        return x

    def age_branch_2(self, inputs, n_classes=2):

        x = self.hidden_layers(inputs)
        
        x = Flatten()(x)
        x = Dense(20)(x)
        x = Activation("relu")(x)
        x = Dense(n_classes)(x)
        x = Activation("softmax", name="age_branch_2")(x)

        return x

    def assemble_model(self, input_shape):

        inputs = Input(shape = input_shape)
        age_branch_1 = self.age_branch(inputs)
        age_branch_2 = self.age_branch_2(inputs)

        model = Model(inputs=inputs,
                      outputs = [age_branch_1, age_branch_2],
                      name="face_net")

        return model

