import numpy as np

X = np.array([
    [3, 5],
    [5, 1],
    [10, 2],

], dtype=float)

Y = np.array([
    [75],
    [82],
    [93],

], dtype=float)

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
Y = Y/100

class NeuralNetworks():

    def __init__(self):
        # build a structure
        self.inputLayerSize = X.shape[1]
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        # random guess
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)

        return yHat

    def sigmoid(self, z):
        # Activation funtion: sigmoid
        return 1/(1+np.exp(-z))


NN = NeuralNetworks()

yHat = NN.forward(X)

print(yHat)

# w=1.0 # random guess
#
# # forward pass
# def forward(x):
#     return w*x
#
# # loss function
# def loss(x, y):
#     yHat = forward(x)
#     return (y-yHat) ** 2.0
#
# # gradient
# def gradient(x, y):
#     return 2*x*(w*x-y)
#
#
#
# # print(np.arange(start=0.0, stop=5.0+0.1, step=0.1))
#
#
# for epoch in range(1, 5+1, 1):
#     for x, y in zip(X, Y):
#         w = w - 0.01 * gradient(x, y)
#         print('---:{}'.format(w))
#         # l = loss(x, y)
#
#     print('Epoch: {}, w={:0.4f}'.format(epoch, w))
#     # print('Epoch: {}, w={:0.4f}, loss={:0.4f}'.format(epoch, w, l))
#
# print('Predict_4:{}'.format(w*4))

# MSE = []
# for w in np.arange(0.0, 5.0, 1.0):
#     # print('w={:.2f}'.format(w))
#
#     mse = 0.0
#     for x, y in zip(X, Y):
#         # yHat = forward(x)
#         mse += loss(x, y)
#     MSE.append((w, mse/3))
#
# def get(i):
#     return i[1]
#
# w = sorted(MSE, key=get)[0][0]
# print(w)
#
#
# import matplotlib.pyplot as plt
# plt.plot(np.arange(0, 5.0, 1), np.array(MSE)[:,1])
# # plt.xlim(0, 1.05)
# # plt.ylim(0, 1.05)
# plt.xlabel('w')
# plt.ylabel('loss')
# plt.title('Loss vs w graph')
# plt.show()


