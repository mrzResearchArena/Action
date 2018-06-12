from keras import Input, Model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

# %%
def getModel():
    inputs = Input(shape=(8,), name='input_layer')
    x = Dense(units=8, init='uniform', activation='relu', name='layer-1')(inputs)
    x = Dense(units=6, init='uniform', activation='relu', name='layer-2')(x)
    outputs = Dense(units=2, init='uniform', name='output_layer', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# %%
def printModel(model):
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='/home/mrz/Desktop/model.jpg')

    print('done')


# %%
import pandas as pd

D = pd.read_csv('/home/mrz/MyDrive/Education/Deep Learning/Keras/diabetes.csv', skiprows=1, header=None).iloc[:,:].values

X = D[:, :-1]
Y = D[:, -1]

Y = to_categorical(Y)

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

# %%
from sklearn.model_selection import train_test_split

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=9)

model = getModel()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=200, batch_size=20, verbose=0)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
