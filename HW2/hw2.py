"""
Kyle Herbruger (Visit my GitHub! https://github.com/Doktorwh0 )
EE 399 HW02
4/18/2023
This program takes in the yalefaces.mat file, which contains 39 faces with 
65 different lighting scenes. 
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image

plt.close('all')
""" ------------------------------ HW2 (a) ------------------------------- """
print("------------------------------ Part a --------------------------------")
"""Generates a plot to show correlation between first 100 faces."""

results = loadmat('yalefaces.mat')
X = np.transpose(results['X']*250)
#im = Image.fromarray(X)
# im.show()

normalizer = 0
corMat = np.zeros((100, 100))
for i in range(100):
    normalizer = 0
    for ii in range(100):
        # Normalizes each image for brightness.
        normi = np.sum(X[ii] * X[ii]) ** 0.5
        normii = np.sum(X[i] * X[i]) ** 0.5
        dotProd = X[i] * X[ii]
        dotProd = (X[i] / normi) * (X[ii] / normii)
        corMat[i, ii] = np.sum(dotProd)

# Normalizes correlation Matrix (corMat) to be 0-1 range.
corMat = corMat / np.max(corMat)
# Plots corMat on a pcolor plot.
plt.figure(1)
plt.title('corMat')
plt.pcolor(corMat, cmap='coolwarm')
plt.colorbar().set_label('Correlation Matrix')

""" ------------------------------ HW2 (b) ------------------------------- """
print("------------------------------ Part b --------------------------------")
corTotal = np.zeros(len(corMat))
for i in range(len(corMat[1])):
    corTotal[i] = np.sum(corMat[i])
plt.figure(2)
plt.title('Face total Correlation')
plt.plot(corTotal)
corBest = corTotal.argmax(axis=0)
corWorst = corTotal.argmin(axis=0)
print('Results:')
print('Best correlated face: ', corBest)
print('Worst correlated face: ', corWorst)

# Showing the faces that correspond the most and least to all the other faces.
im = Image.fromarray(
    np.concatenate((X[corBest].reshape(32, 32),
                   X[corWorst].reshape(32, 32)))
)
im.show()
# im = Image.fromarray(X[corWorst].reshape(32, 32))
# im.show()
# Results are reasonable, as a blank image has little detail to correspond with.
# Meanwhile, the best seems to have slight upwards, but directly frontal,
# lighting which shows the details of the face well, allowing it to correspond
# well with other faces, or at least partially with partially lit faces.

""" ------------------------------ HW2 (c) ------------------------------- """
print("------------------------------ Part c --------------------------------")

imgsOfInt = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
corMat_2 = np.zeros((10, 10))
for i in range(len(imgsOfInt)):
    for ii in range(len(imgsOfInt)):
        normi = 0
        normii = 0
        dotProd = X[imgsOfInt[i]] * X[imgsOfInt[ii]]
        for iii in range(len(X[1])):
            normii = X[imgsOfInt[ii], iii] * X[imgsOfInt[ii], iii] + normii
            normi = X[imgsOfInt[i], iii] * X[imgsOfInt[i], iii] + normi

        normi = normi ** 0.5
        normii = normii ** 0.5
        dotProd = (X[imgsOfInt[i]] / normi) * (X[imgsOfInt[ii]] / normii)
        corMat_2[i, ii] = np.sum(dotProd)

corMat_2 = corMat_2 / np.max(corMat_2)
plt.figure(3)
plt.title('corMat_2')
plt.pcolor(corMat_2, cmap='coolwarm')
plt.colorbar().set_label('Correlation Matrix')

""" ------------------------------ HW2 (d) ------------------------------- """
print("------------------------------ Part d --------------------------------")

Y = np.dot(X, X.T)

# compute the eigenvalues and eigenvectors of Y
eigenvalues, eigenvectors = np.linalg.eig(Y)

# sort the eigenvectors by their corresponding eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

# select the first six eigenvectors
eigenvectors = eig6 = eigenvectors[:, :6].T

# print the shape of the eigenvectors
print(eigenvectors.shape)

""" ------------------------------ HW2 (e) ------------------------------- """
print("------------------------------ Part e --------------------------------")

# compute the SVD of X
U, s, Vt = np.linalg.svd(X.T, full_matrices=False)

# extract the first six columns of the matrix of right singular vectors
principal_components = svd6 = Vt[:6, :]

# print the shape of the principal components
print(principal_components.shape)

""" ------------------------------ HW2 (f) ------------------------------- """
print("------------------------------ Part f --------------------------------")
norm_diff = (np.sum((abs(eig6) - abs(svd6)) ** 2)) ** 0.5
print("Normalized Difference: ", norm_diff)
print("                                         AKA, reallllllllllllly small.")

""" ------------------------------ HW2 (g) ------------------------------- """
print("------------------------------ Part g --------------------------------")
print()
perc_var = np.ones((6, 2414))
svd_var = np.ones((6, 2414))
for i in range(6):
    svd_var[i] = np.ones(len(svd6[1])) * np.mean(svd6[i])
    svd_var[i] = (svd6[i] - svd_var[i]) ** 2
    perc_var[i, 1] = 1/2414 * np.sum(svd_var[i])
    perc_var[i] = svd6[i] / perc_var[i, 1] * 100
perc_var_sum = np.ones(6)
offset = np.ones(2414)
for i in range(6):
    svd6[i] = svd6[i] + offset * i * 0.3
    perc_var_sum[i] = -1 * np.sum(perc_var[i])
plt.figure(5)
plt.title('SVD Modes')
plt.plot(svd6[0])
plt.plot(svd6[1])
plt.plot(svd6[2])
plt.plot(svd6[3])
plt.plot(svd6[4])
plt.plot(svd6[5])

plt.figure(6)
plt.title('SVD vector variance')
plt.plot(perc_var_sum)
