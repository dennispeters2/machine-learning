# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 23:24:29 2019
Modified on Jan 30 2023 106am

@author: Dennis Peters
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

# %matplotlib inline            # don't forget this if you're using jupyter!
# C:\Users\User\Desktop\SDS ML\data

# local file
print(os.getcwd())
os.chdir(os.path.dirname("C:\\Users\\user\\Desktop\\SDS ML\\data\\"))
print(os.getcwd())

X = pd.read_csv("titanic_train.csv")
X = X[['Pclass', 'Age', 'Fare', 'Parch', 'SibSp', 'Survived']].dropna()
y = X.pop('Survived')

model = RandomForestClassifier()
model.fit(X, y)


print(plt.rcParams.get('figure.figsize'))
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 6

(pd.Series(model.feature_importances_, index=X.columns)
   .nlargest(4)
   .plot(kind='barh'))        # some method chaining, because it's sexy!


########################################################
# dennis like this one! 
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])
for name, importance in zip(iris["feature_names"], rnd_clf.feature_importances_):
  print(name, "=", importance)

features = iris['feature_names']
importances = rnd_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
########################################################


from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create decision tree classifer object
clf = RandomForestClassifier(random_state=0, n_jobs=-1)
# Train model
model = clf.fit(X, y)

# Calculate feature importances
importances = model.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [iris.feature_names[i] for i in indices]

# Barplot: Add bars
plt.bar(range(X.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=20, fontsize = 8)
# Create plot title
plt.title("Feature Importance")
# Show plot
plt.show()


