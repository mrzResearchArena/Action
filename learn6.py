import glob

PATH = '/home/mrz/MyDrive/Education/Deep Learning/numta'

x_train = sorted(glob.glob('/'.join([PATH, 'training-?', '*.png'])))
y_train = sorted(glob.glob('/'.join([PATH, 'training-?.csv'])))

