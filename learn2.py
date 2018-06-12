PATH = '/home/mrz/MyDrive/Entertainment/myPicture/mrz.jpg'

from keras.preprocessing.image import load_img, img_to_array, array_to_img

reduce = 4
#
v = load_img(PATH, target_size=(2, 4))
#
# # v = img_to_array(PATH)
# print(v.shape)
#
import numpy as np
#
#



#
import matplotlib.pyplot as plt
plt.imshow(v)
plt.show()

print(np.array(v))

# v = np.array([
#     [[[1, 2, 3],[4, 5, 6],[7, 8, 9]],
#     [[10, 20, 30],[40, 50, 60],[70, 80, 90]],
#     [[100, 200, 300],[400, 500, 600],[700, 800, 900]],
#     [[10000, 20000, 30000],[40000, 50000, 60000],[70000, 80000, 90000]]],
#
#     [[[11, 2, 3],[4, 5, 6],[7, 8, 9]],
#     [[10, 20, 30],[40, 50, 60],[70, 80, 90]],
#     [[100, 200, 300],[400, 500, 600],[700, 800, 900]],
#     [[10000, 20000, 30000],[40000, 50000, 60000],[70000, 80000, 90000]]],
# ])
#
# print(v.shape) # (2,4,3,3)
# print(v.ndim)  # 4D
#
# print(v[1][0][0][0])