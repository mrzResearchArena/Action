from keras.models import Input, Model
from keras.activations import relu, softmax
from keras.layers import Dense

from PIL import Image
# %%

# from keras.datasets import mnist, imdb, cifar10, cifar100, reuters, fashion_mnist, boston_housing
# datasets = [mnist, imdb, cifar10, cifar100, reuters, fashion_mnist, boston_housing]
#
# for i in datasets:
#     i.load_data()
#
#     print('-- done --')


# # %%
#
# def getModel():
#     inputs = Input(shape=(28, 28))
#     x = Dense(units=512, activation=relu)(inputs)
#     outputs = Dense(units=10, activation=softmax)(x)
#
#     model = Model(inputs=inputs, outputs=outputs)
#
#     return model
#
#
# model = getModel()
#
#
# # %%
# def printModelInline(model):
#     from keras.utils import plot_model
#     plot_model(model, show_shapes=True, to_file='model.jpg')
#
#     return Image.open('model.jpg')
#
#
# image = printModelInline(model)
# # %%
#
# image.show()

# import numpy as np
#
# v = np.array([
#     [[45, 78],[41, 1], [12, 32]]
# ])
# print(v.ndim)

import numpy as np

a = np.array([
    [2, 3, 4],
    [1, 2, 6],
])

b = np.array([
    [1, 4],
    [2, 2],
    [3, 1],
])

result = np.dot(a, b) + np.array([
    [2, 2],
    [2, 3],
])

def sigmoid(x):
    return 1.0 / (1+np.exp(x))

# result = sigmoid(result)

v = np.array([
    [1.2, -0.9, 0.4],
    [100, 14, 91],
    [20, 9, -81],
])


x = np.array([
    [1, 2, 3],
    [4, 5, 6],
])


y = np.array([
    [10, 20],
    [30, 40],
    [50, 60],
])

# def dotOwn(x, y):
#     assert x.shape[1] == y.shape[0], 'Check matrix dimension'
#
#     x = x.copy()
#     y = y.copy()
#
#     result = np.empty(shape=(x.shape[0], y.shape[1]))
#
#     for row in range(x.shape[0]):
#         for column in range(y.shape[1]):
#             result[row, column] = np.sum(x[row,:] * y[:,column])
#
#     return result
#
# print(dotOwn(x, y))
#
# print(np.sum(np.array([1, 2, 3] * np.array([10, 30, 50]))))
#

def tp(v):
    v = v.copy()
    result = np.zeros(shape=(v.shape[1], v.shape[0]))

    for row in v:
        print(row)

v = np.array([
    [10, 20, 30],
    [5,  10, 15],
])
print(tp(v))

import pydicom as v
