import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

csv_file = 'emnist-balanced-train.csv'
cont = pd.read_csv(csv_file)
arr = np.array(cont)
# print(arr.shape) # (112799, 785)
y = arr[:, 0]
# X = arr[:, 1:].reshape((112799, 28, 28))
X = arr[:, 1:] / 255


# trim 0~9
y_trim = []
X_trim = []
for i in range(y.shape[0]):
	if y[i] >= 10 and y[i] <= 35: 
		y_trim.append(y[i])
		X_trim.append(X[i, :])
y_trim = np.array(y_trim)
# print(y_trim.shape) # (62400, )
X_trim = np.stack(X_trim, axis=0)
# print(X_trim.shape) # (62400, 784)
noisy_X = X_trim + np.random.normal(0,0.25,X_trim.shape)

col = 5
row = 5
f, a = plt.subplots(row, col, figsize=(col, row))
for i in range(row):
	for j in range(col):
		a[i][j].imshow(noisy_X[i*row + j, :].reshape((28,28)).transpose(), cmap='gray')
		# print(y_trim[i*row + j])
plt.show()

f = open('data.pickle', 'wb')
pickle.dump((X_trim, noisy_X, y_trim), f)
f.close()