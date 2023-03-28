from cProfile import label
from sys import stderr
from turtle import pd
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy.stats import linregress
from scipy.stats import t

"""
Import the dataset
"""

df = pd.read_excel("Data.xlsx")

"""
split data in x and y values
"""

x = df["x"].values.reshape(-1,1)
y = df["y"].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
ax1.plot(x, y, linestyle = "dotted", label="Dataset")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.legend(loc="upper left")


"""
Split data into training and test set
"""
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

"""
Make linear regression using sklearn
"""
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
mean_prediction = reg.predict(x)

"""
Create regression using scipy
"""
model = linregress(X_train.reshape(-1), y_train)
reg2 = model.slope * x + model.intercept

"""
Plot the results
"""
ax2.plot(x[0:len(X_train)], y[0:len(X_train)], linestyle="dotted", label="Training set") # plot Training set
ax2.plot(x[len(X_train):], y[len(X_train):], linestyle = "dotted", label = "Test set")  # Plot test set
ax2.plot(x, mean_prediction, label= "Regression model") # Plot regression model sklearn
ax2.plot(x, reg2, linestyle='dashed', label= "Regression with scipy")   # plot regression model scipy
ax2.set_xlabel("$x$")   # x label
ax2.set_ylabel("$y$")   # y label
ax2.legend(loc="upper right")   # legend    


"""
Show graphs
"""
plt.show()