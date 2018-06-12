# import numpy as np
#
# v = np.zeros(shape=(2,4,3))
#
# print(v.shape)
#
# print(v)
#
#
# import matplotlib.pyplot as plt
# plt.imshow(v)
# plt.show()

PATH = '/home/mrz/MyDrive/Entertainment/myPicture/mrz.jpg'
import numpy as np
import matplotlib.pyplot as plt

# v = np.zeros(shape=(2, 4, 3))

# v[0][0][0] = 0
# v[0][0][1] = 0
# v[0][0][2] = 0

# v[0][1][0] = 128
# v[0][1][1] = 128
# v[0][1][2] = 128

# v[0][1][0:3] = 128

# v[0][2][0] = 0
# v[0][2][1] = 0
# v[0][2][2] = 0

# v[0][3][0] = 128
# v[0][3][1] = 128
# v[0][3][2] = 128

# v[0][3][0:3] = 128


# v[1][0][0] = 128
# v[1][0][1] = 128
# v[1][0][2] = 128

# v[1][0][0:3] = 128

# v[1][2][0] = 128
# v[1][2][1] = 128
# v[1][2][2] = 128

# v[1][2][0:3] = 128

# import numpy as np
# x = np.ones((3,3))
# print("Checkerboard pattern:")
# x = np.zeros((8,8),dtype=int)
# x[1::2,::2] = 1
# x[::2,1::2] = 1
# print(x)

v = np.zeros(shape=(2,4))

v[0][0] = 1
v[0][2] = 1
v[1][1] = 1
v[1][3] = 1


plt.imshow(v)
plt.show()