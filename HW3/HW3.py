"""
Kyle Herbruger (Visit my GitHub! https://github.com/Doktorwh0 )
EE 399 HW03
4/24/2023

"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# %%
plt.close('all')
""" ------------------------------ HW3 (1) ------------------------------- """
print("------------------------------ Part a --------------------------------")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape[0])
print(x_test.shape)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
print(x_train_flat.shape)
for i in range(6):
    plt.figure(6 + i)
    plt.imshow(x_train[i])

# %%
# Perform SVD analysis on the flattened training data
normalizer = 0
corMat = np.zeros((100, 784))
for i in range(100):
    normalizer = 0
    for ii in range(100):
        # Normalizes each image for brightness.
        normi = np.sum(x_train_flat[ii] ** 2) ** 0.5
        normii = np.sum(x_train_flat[i] ** 2) ** 0.5
        dotProd = x_train_flat[i] * x_train_flat[ii]
        dotProd = (x_train_flat[i] / normi) * (x_train_flat[ii] / normii)
        corMat[i, ii] = np.sum(dotProd)

# Normalizes correlation Matrix (corMat) to be 0-1 range.
corMat = corMat / np.max(corMat)
U, s, Vt = np.linalg.svd(corMat.T, full_matrices=False)

# %%
print('U: ', U.shape)
print('s: ', s.shape)
print('Vt: ', Vt.shape)
print('Vt.T: ', Vt.T.shape)

# Plot the singular value spectrum
plt.figure(1)
plt.plot(s)
plt.title("Singular Value Spectrum")
plt.xlabel("Mode")
plt.ylabel("Singular Value")
plt.show()

# Determine the number of modes necessary for good image reconstruction
s_energy = np.cumsum(s ** 2)
s_total_en = np.sum(s ** 2)
s_step_en = s_energy / s_total_en
num_modes = np.sum(s_step_en < 0.90) + 1
print("Number of modes for 90% energy retention:", num_modes)

# %%
plt.close('all')
plt.figure(2)
plt.plot(s[:num_modes])
plt.show()
plt.figure(3)
plt.plot(Vt[1])
plt.show()
plt.figure(4)
plt.plot(Vt.T[0])
plt.plot(Vt.T[1] - 1)
plt.plot(Vt.T[2] - 2)
plt.plot(Vt.T[3] - 3)
plt.show()


# U, S, and V matrices.
# U is the eigenvalue matrix, and represents the unique fa
