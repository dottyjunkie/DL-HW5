import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

csv_file = 'emnist-balanced-train.csv'
cont = pd.read_csv(csv_file)
arr = np.array(cont)
# print(arr.shape) # (112799, 785)
y = arr[:, 0]
X = arr[:, 1:].reshape((112799, 28, 28))

col = 5
row = 5
f, a = plt.subplots(row, col, figsize=(col, row))
for i in range(row):
	for j in range(col):
		sample = X[i*row + j, :, :].transpose()
		noise = np.random.normal(0,64,(28,28))
		a[i][j].imshow(sample + noise, cmap='gray')
plt.show()

f = open('data.pickle', 'wb')
pickle.dump((X, y), f)
f.close()