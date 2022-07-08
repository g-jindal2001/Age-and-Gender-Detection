import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Activation, Dense, MaxPool2D, GlobalAveragePooling2D, Flatten

class AgeDetector(tf.keras.Model):
    def __init__(self):
        super(AgeDetector, self).__init__()

        self.conv1 = Conv2D(filters=8, kernel_size=3)
        self.actv1 = Activation("relu")
        self.conv2 = Conv2D(filters=16, kernel_size=3)
        self.actv2 = Activation("relu")
        self.conv3= Conv2D(filters=32, kernel_size=3)
        self.actv3 = Activation("relu")
        self.pool1 = MaxPool2D(pool_size=(2,2))

        self.conv4 = Conv2D(filters=32, kernel_size=3)
        self.actv4 = Activation("relu")
        self.pool2 = MaxPool2D(pool_size=(2,2))

        self.conv5 = Conv2D(filters=64, kernel_size=3)
        self.actv5 = Activation("relu")
        self.pool3 = MaxPool2D(pool_size=(2,2))

        self.flatten = GlobalAveragePooling2D()
        self.dense1 = Dense(20)
        self.actv6 = Activation("relu")
        self.dense2 = Dense(7)
        self.actv7 = Activation("softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.actv1(x)
        x = self.conv2(x)
        x = self.actv2(x)
        x = self.conv3(x)
        x = self.actv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.actv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.actv5(x)
        x = self.pool3(x)
    
        print(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.actv6(x)
        x = self.dense2(x)
        x = self.actv7(x)

        return x

