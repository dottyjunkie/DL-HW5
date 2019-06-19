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
noisy_X = X + np.random.normal(0,0.25,(112799, 784))

col = 5
row = 5
f, a = plt.subplots(row, col, figsize=(col, row))
for i in range(row):
	for j in range(col):
		a[i][j].imshow(noisy_X[i*row + j, :].reshape((28,28)).transpose(), cmap='gray')
plt.show()

f = open('data.pickle', 'wb')
pickle.dump((X, noisy_X, y), f)
f.close()