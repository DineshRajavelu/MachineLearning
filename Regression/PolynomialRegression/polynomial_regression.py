
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/devstack/Documents/MachineLearning/Regression/PolynomialRegression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


# Fitting Polynomial Regression to the Training set
from sklearn.preprocessing import  PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

# Visualizing the Linear Regression Model
plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''

#Visualizing the Polynomial Regression Model
# X_grid = np.arange(min(X),max(X),0.1)
# X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''
# Predicting the new result with Linear Regression
lin_reg.predict(6.5)

# Predicting the new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
