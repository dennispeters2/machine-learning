# Simple Linear Regression

# DPeters 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor


import os
print(os.getcwd())

os.chdir(os.path.dirname("C:\\Users\\User\\Desktop\\SDS ML\\Simple_Linear_Regression\\Simple_Linear_Regression\\"))
print(os.getcwd())

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print(dataset.head(3))
print(dataset.tail(3))

print(dataset.shape)

print(dataset.describe())

# print('mean')
# print(dataset.mean())
print('   median')
print(dataset.median(axis='rows'))
print('   mode')
print(dataset.mode(numeric_only=True, dropna=False))


# Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression # Ctrl + 1 gives you info 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print('Ultimately, we want to compare Y_TEST real salaries to Y_PRED predicted salaries!')

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print('Accuracy:')
print(regressor.score(X,y))

print('Get_params:')
print(regressor.get_params())

# The coefficients
rc = regressor.coef_
print('Coefficients: \n', str(rc))
print('Each year of experience is rougly worth $' + str(rc))


ri = regressor.intercept_
print('Intercept: \n', regressor.intercept_)
print('0 years of experience = $' + str(ri))


from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error
# print("Mean squared error: %.2f"
 #      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print('we want variance score close to 1, < .35 is bad, >.60 is decent')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# MSE with numpy 
# MSE = np.square(np.subtract(y_test, y_pred)).mean()
# import math
# math.sqrt(MSE)


'''
great example
http://benalexkeen.com/linear-regression-in-python-using-scikit-learn/

https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
'''

#####################################

import statsmodels.api as sm

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


