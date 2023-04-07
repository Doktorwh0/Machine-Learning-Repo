"""
Kyle Herbruger (Visit my GitHub! https://github.com/Doktorwh0 )
EE 399 HW01
4/7/2023
Best fit function calculator.
This program solves 4 assigned tasks for EE399 HW1. These generally involve
fitting functions to data to learn about overfitting.
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import warnings

#============================================================================#
# Calculates the y data for the Asin(Bx) + Cx + D function with the given
# paramters and x data set.


def func(x, A, B, C, D):
    return A * np.sin(B * x) + C * x + D

#============================================================================#
# Returns an array with the difference between the fit function with the given
# params, and the difference between each point of the given data set.
# Returns an array.


def objective(params, x, y):
    A, B, C, D = params
    return y - func(x, A, B, C, D)

#============================================================================#
# Finds the difference between the fit function based on the given parameters
# and the given data set. Returns least squares error.
# TODO! change code to use least_sqrs_dif function for part (ii)


def objective_dif(params, x, y):
    A, B, C, D = params
    diff = y - func(x, A, B, C, D)
    diff = diff * diff
    diff = diff / len(x)
    difSum = np.sum(diff)
    difSum = pow(difSum, 0.5)
    return difSum

#============================================================================#
# Calculates the LSE for a fit function and data set.
# Input:
#   fitFuncData: array of y values of the fit function.
#             x: array of x values from the data set.
#             y: array of y values from the data set.
# Returns Least Square Error.


def least_sqrs_dif(fitFuncData, x, y):
    diff = y - fitFuncData
    diff = diff * diff
    diff = diff / len(x)
    difSum = np.sum(diff)
    difSum = pow(difSum, 0.5)
    return difSum

#============================================================================#
# Plots the LSE for varied A-D parameters on the sin function used in part i.
# Input:
#     xAxis: xAxis label given as string.
#     yAxis: yAxis label given as string.
#    figNum: the figure number to use.
#   figData: The data to plot. Should be 2D array.
# Outputs a plot.


def pcolor_plot(xAxis, yAxis, figNum, figData):
    plt.figure(figNum)
    plt.pcolor(figData, cmap='coolwarm')
    plt.ylabel(xAxis)
    plt.xlabel(yAxis)
    plt.colorbar()


"""            ***********      Start of main      ***********            """
print("Least Squares Error is abbreviated to LSE.")
# Given data set from lecture.
xData = np.arange(31)
yData = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45,
                  41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
""" ------------------------------ HW1 (i) ------------------------------ """
print("------------------------------ Part i -------------------------------")
# Initial parameters to test with.
params0 = [1, 1, 1, 1]
# Fitting using least squares.
result = least_squares(objective, params0, args=(xData, yData))

# Outputting results.
print("Best fit parameters:")
print('                    A =', round(result.x[0], 2))
print('                    B =', round(result.x[1], 2))
print('                    C =', round(result.x[2], 2))
print('                    D =', round(result.x[3], 2))
print("Best fit LSE: ", round(objective_dif(result.x[0:4], xData, yData), 2))

# Plotting given data set as well as the best fit function.
plt.close('all')
plt.figure(1)
plt.scatter(xData, yData, label='Data')
plt.plot(xData, func(xData, *result.x), label='Fit Curve')
plt.legend()
plt.show()

""" ------------------------------ HW1 (ii) ----------------------------- """
print("------------------------------ Part ii ------------------------------")
print("Refer to figures 2-7.")
# The following sections vary each parameter of the sin function by -25 to 25.
# Amount to vary parameters by.
sweepRange = 25
# CD -------------------------------------------------- #
dataArray = np.zeros((sweepRange*2, sweepRange*2))
for i in np.arange(-sweepRange, sweepRange):
    for ii in np.arange(-sweepRange, sweepRange):
        dataArray[i+sweepRange, ii+sweepRange] = objective_dif([result.x[0],
                                                                result.x[1],
                                                                result.x[2] + i,
                                                                result.x[3] + ii],
                                                               xData, yData)
pcolor_plot('C', 'D', 2, dataArray)

# AB -------------------------------------------------- #
dataArray = np.zeros((sweepRange*2, sweepRange*2))
for i in np.arange(-sweepRange, sweepRange):
    for ii in np.arange(-sweepRange, sweepRange):
        dataArray[i+sweepRange, ii+sweepRange] = objective_dif([result.x[0] + i,
                                                                result.x[1] + ii,
                                                                result.x[2],
                                                                result.x[3]],
                                                               xData, yData)
pcolor_plot('A', 'B', 3, dataArray)

# AC -------------------------------------------------- #
dataArray = np.zeros((sweepRange*2, sweepRange*2))
for i in np.arange(-sweepRange, sweepRange):
    for ii in np.arange(-sweepRange, sweepRange):
        dataArray[i+sweepRange, ii+sweepRange] = objective_dif([result.x[0] + i,
                                                                result.x[1],
                                                                result.x[2] + ii,
                                                                result.x[3]],
                                                               xData, yData)
pcolor_plot('A', 'C', 4, dataArray)

# BC -------------------------------------------------- #
dataArray = np.zeros((sweepRange*2, sweepRange*2))
for i in np.arange(-sweepRange, sweepRange):
    for ii in np.arange(-sweepRange, sweepRange):
        dataArray[i+sweepRange, ii+sweepRange] = objective_dif([result.x[0],
                                                                result.x[1] + i,
                                                                result.x[2] + ii,
                                                                result.x[3]],
                                                               xData, yData)
pcolor_plot('B', 'C', 5, dataArray)

# AD -------------------------------------------------- #
dataArray = np.zeros((sweepRange*2, sweepRange*2))
for i in np.arange(-sweepRange, sweepRange):
    for ii in np.arange(-sweepRange, sweepRange):
        dataArray[i+sweepRange, ii+sweepRange] = objective_dif([result.x[0],
                                                                result.x[1],
                                                                result.x[2] + i,
                                                                result.x[3] + ii],
                                                               xData, yData)
pcolor_plot('A', 'D', 6, dataArray)

# BD -------------------------------------------------- #
dataArray = np.zeros((sweepRange*2, sweepRange*2))
for i in np.arange(-sweepRange, sweepRange):
    for ii in np.arange(-sweepRange, sweepRange):
        dataArray[i+sweepRange, ii+sweepRange] = objective_dif([result.x[0],
                                                                result.x[1] + i,
                                                                result.x[2],
                                                                result.x[3] + ii],
                                                               xData, yData)
pcolor_plot('B', 'D', 7, dataArray)

""" ----------------------------- HW1 (iii) ----------------------------- """
print("----------------------------- Part iii ------------------------------")
# Fitting a 19th degree polynomial to the data using the first 20 data points.
xDataTrain = xData[0:21]
yDataTrain = yData[0:21]
xDataTest = xData[20:31]
yDataTest = yData[20:31]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    coeffs = np.polyfit(xDataTrain, yDataTrain, 19)
p = np.poly1d(coeffs)

plt.figure(8)
plt.ylim(0, 60)
plt.scatter(xDataTrain, yDataTrain)
plt.plot(xDataTrain, p(xDataTrain))
print("Part iii Training LSE: ",
      round(least_sqrs_dif(p(xDataTrain), xDataTrain, yDataTrain), 2))

# Testing the fit function from the first 20 points.
plt.figure(9)
plt.yscale('log', base=100)
# Should be noted the original data set is not differentiable from a flat line
# even with a base 1000 log scale for the y-axis. Defaulted to base 100 due
# to base 1000 lagging.
plt.scatter(xDataTest, yDataTest)
plt.plot(xDataTest, p(yDataTest))
print("Part iii Test LSE: ",
      round(least_sqrs_dif(p(xDataTest), xDataTest, yDataTest), 2))

""" ------------------------------ HW1 (iv) ----------------------------- """
print("------------------------------ Part iv ------------------------------")
# Repeats the previous section, but with the first 10 and last 10 data points.
# Tests on the middle 10.
xDataTrain = np.concatenate((xData[0:11], xData[20:31]))
yDataTrain = np.concatenate((yData[0:11], yData[20:31]))
xDataTest = xData[10:21]
yDataTest = yData[10:21]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    coeffs = np.polyfit(xDataTrain, yDataTrain, 19)
p2 = np.poly1d(coeffs)

plt.figure(10)
plt.ylim(0, 60)
plt.scatter(xDataTrain, yDataTrain)
plt.plot(xDataTrain, p2(xDataTrain))
print("Part iv Training LSE: ",
      round(least_sqrs_dif(p2(xDataTrain), xDataTrain, yDataTrain), 2))

plt.figure(11)
plt.scatter(xDataTest, yDataTest)
plt.plot(xDataTest, p2(xDataTest))
print("Part iv Test LSE: ",
      round(least_sqrs_dif(p2(xDataTest), xDataTest, yDataTest), 2))
