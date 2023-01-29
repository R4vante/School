import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF
from sklearn.kernel_ridge import KernelRidge


def generate_dataset(N,noiseSTD):
    """for given list length and standard deviation, create a vector of length N and calculate y values of given function

    Args:
        N (integer): list length
        noiseSTD (float): standard deviation

    Returns:
        Return value is object with following attributes:
            x: list of x values and y values
            y: list of y values
    """   

    x = np.linspace(0, 4*np.pi, N)
    y = 2 - np.sin(x) - 0.5*x + np.exp(x/6) + noiseSTD*np.random.randn(len(x))
    return x, y


# Import excel spreadsheet of given training set

df = pd.read_excel('Data.xlsx')

# x, y = generate_dataset(12, 500, 0.4)
# x = x.reshape(-1,1)

""" Split colomns in x and y values
    x vector has to be reshapen to make it a array. The -1 in reshape() means the dimension of the vector is unknow and numpy will figured it out
    by it's own."""

x = df['x']
x = x.values.reshape(-1,1)
y = df['y'].values


""" Split the x and y values in a training and test set.
    -Training set is 80%
    -Test set is 20% """

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


""" To do periodic regression -> use Gaussian process regressor.
    Set kernel to radius basis function (RBF()) with white noise (Whitekernel).
    White noise implies that the data contains noise. 
    
    n_restarts_optimizer is the number of restarts of the optimizer to find the kernel's parameters."""
kernel = 1.0 * RBF() + WhiteKernel(1e-1)
gaussian_process= GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
gaussian_process.fit(X_train, y_train)


""" Extract following attributes from the gaussian_process object:
    - mean_prediction: prediction of the mean value of given dataset.
    - std_prediction: prediction of the standard deviation of given datset. """

mean_prediction, std_prediction = gaussian_process.predict(x, return_std=True)


""" Plot the results """
fig, ax = plt.subplots(1)

# Scatter plot of training and test set.
ax.plot(x[0:len(X_train)],y[0:len(X_train)], linestyle='dotted', label="Training set")
ax.plot(x[len(X_train):],y[len(X_train):], linestyle='dotted', label="Test set")

#Plot the mean prediction of the regression
ax.plot(x, mean_prediction, label="Mean prediction")

# Make a fill to indicate standard deviation.
ax.fill_between(
    x.ravel(),
    mean_prediction -  std_prediction,
    mean_prediction + std_prediction,
    alpha = 0.5, 
    label = r"95% confidence interval",
)


# Set labels and legend
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(loc = "upper left")


""" Create a new dataset with the premade function generate_dataset().
    This to test if made regression works for a different dataset. """

x_new, y_new = generate_dataset(750, 0.4)
x_new = x_new.reshape(-1,1)

""" With the new data set, calculate new regression line with the pre-calculated regression parameters.
    Extract from the gaussian_process object following attributes:
    - mean_pred2: mean prediction of given dataset.
    - std_pred2: standard deviation prediction of given dataset. """

mean_pred2, std_pred2 = gaussian_process.predict(x_new, return_std=True)

""" Create second plot for given regression. """

fig2, ax2 = plt.subplots(1,1)
# scatter plot of new dataset
ax2.plot(x_new, y_new, linestyle='dotted', label='Observations')

# Plot of regression
ax2.plot(x_new, mean_pred2, label="Mean prediction", color='green')

# Fill of confidence level
ax2.fill_between(
    x_new.ravel(),
    mean_pred2 - std_pred2,
    mean_pred2 + std_pred2,
    alpha=0.5,
    color='red',
    label=r"95% confidence interval",
)

# labels and legend
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.legend(loc = "upper left")


# show figures
plt.show()