import keras
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D
# import numpy as np


batch_size = 512
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


def getModel():
    inputs = Input(shape=(28,28,1))
    x = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1))(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = getModel()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


