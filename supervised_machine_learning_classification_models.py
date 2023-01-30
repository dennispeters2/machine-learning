# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 22:37:50 2018
Last Modified Jan 28, 2023 - supress future warnings from sci kit learn

@author: Dennis Peters

1st in the series
SUPERVISED LEARNING CLASSIFICATION MODELS

IRIS
also pima indian feature extraction 

Techniques: 
EDA
plots
correlation matrix
train test
kfold
make predictions
feature power

"""

# =============================================================================
# # Check the versions of libraries
# 
# # Python version
# import sys
# print('Python: {}'.format(sys.version))
# # scipy
# import scipy
# print('scipy: {}'.format(scipy.__version__))
# # numpy
# import numpy
# print('numpy: {}'.format(numpy.__version__))
# # matplotlib
# import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# # pandas
# import pandas
# print('pandas: {}'.format(pandas.__version__))
# # scikit-learn
# import sklearn
# print('sklearn: {}'.format(sklearn.__version__))
# 
# =============================================================================

# Load libraries
import numpy as np

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pandas.plotting import scatter_matrix

import os
import seaborn as sns

# import urllib.request

# from sklearn.model_selection import train_test_split
from sklearn import datasets
# from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# ctrl + 1 for comments
# ctrl + L 
# F9

# Load dataset
# URL
'''
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
'''
# shape
print(' ')

# local file
print(os.getcwd())
os.chdir(os.path.dirname("C:\\Users\\user\\Desktop\\SDS ML\\data\\"))
print(os.getcwd())
dataset = pd.read_csv('iris2.csv')


print(dataset.shape)
print(' ')
# head
print(dataset.head(5))
print(' ')
# descriptions
print(dataset.describe())
print(' ')
# class distribution
print(dataset.groupby('class').size())
print(' ')

print(dataset['class'].value_counts())
print(' ')

print(dataset['petal-width'].value_counts())
print(' ')


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(7,7))
plt.show()

# Same as the above, but with `sharex` and `sharey` set to True. 
# Helps me understand the relative size of the attributes 

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=True, sharey=True, figsize=(10,10))
plt.show()

# histograms
dataset.hist(figsize=(15,10))
plt.show()

"""
 scatter plot matrix
# Diagonal grouping of some attribute-pairs indicate high correlation and 
# predictable relationship.
 """
scatter_matrix(dataset, figsize=(15,10))
plt.show()

# PRETTY 
# %matplotlib inline
import seaborn as sns; sns.set()
sns.pairplot(dataset, hue='class', size=2);


"""
# $$$ BEST CORRELATION chart  https://github.com/Jimsparkle/bitcoin/blob/master/Correlation.ipynb
"""
import seaborn as sns

sns.set(font_scale=1.0)
fig, ax = plt.subplots(figsize=(5,5)) 
sns.heatmap(dataset.corr(min_periods=12), annot=True, annot_kws={"size": 12}, fmt='.1f')
plt.show()


# Prepare Test Traing for Modelling
# Split-out validation dataset
# 120 and 30. 4 (features) and 1 (class)
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


print(dataset.shape) # 150

print(X_train.shape) 
print(Y_train.shape)
print(X_validation.shape)
print(Y_validation.shape)


# BUILD A MODEL -  Support Vector Classifier
# SVM
print('SVC()')
# Build an SVC model for predicting iris classifications using training data

mymod = SVC()
# mymod = LogisticRegression()
mymod.fit(X_train, Y_train)
# predictions = mymod.predict(X_validation)

# Now measure its performance with the test data
# what is the accuracy of my model 
mymod.score(X_validation, Y_validation)  

# print(my_mod)


'''

MAKE SPOT PREDICTIONS

'''


# SPOT CHECKS
# print 1st 5 rows of iris untouched
print(dataset.head(5))

# make new array 
z = array[:,0:4]
print(z.shape)
# takes first 4 rows- ALL cols
z2 = z[0:4,]
z2
# below is take rows 2 and 3
z2 = z[1:3,]
z2

# feed the model the array of numbers and model will return the classification
print(mymod.predict(z2))

print(dataset.tail(5))
# takes last 3 rows- ALL cols
z3 = z[147:151,]
print(z3)

print(mymod.predict(z3))



# feed in some values and make a prediction 
X_new = np.array([[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]])
print("X_new.shape: {}".format(X_new.shape))
X_new.shape: (2, 4)
prediction = mymod.predict(X_new)
# 2.1 Test result prediction

#Prediction of the species from the input vector
print("Prediction of Species: {}".format(prediction))


#########################################################
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms for accuracy
models = []
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10, random_state=1)

# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()

# DP add 1/28/23 supress warnings 
# https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# evaluate each model in turn
# 10 fold cross validation
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    

# Compare Algorithms THIS DOES NOT WORK 
    '''
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
''' 


"""
# $$$ Make predictions on validation dataset
LDA was winner with 97% !
KNN gives best study case for confusion matrix!!!! 
"""
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print("KNN Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print()

print("confusion_matrix:")
print(confusion_matrix(Y_validation, predictions))
print()

'''
https://tatwan.github.io/How-To-Plot-A-Confusion-Matrix-In-Python/
'''

print("classification_report:")
print(classification_report(Y_validation, predictions))
print()

'''
https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
precision = TP / TP + FP  = What proportion of positive identifications was  correct?
recall    = TP / TP + FN  = What proportion of ACTUAL positives was  correct?
F1 = a balance of precision and recall
'''

print(Y_validation) # this is the actual vlaue 
print(predictions) # this is the prediction dennis made - 3 were wrong 


print('/n pass in values and make a prediction')
from sklearn import neighbors, datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
print(iris.target_names[knn.predict([[3, 5, 4, 2]])])
print()


# SVM
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)

print("SVC Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print()

print("confusion_matrix:")
print(confusion_matrix(Y_validation, predictions))
print()

print("classification_report:")
print(classification_report(Y_validation, predictions))
print()


print('LogisticRegression')
mymod = LogisticRegression()
mymod.fit(X_train, Y_train)
predictions = mymod.predict(X_validation)

print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print()

print("confusion_matrix:")
print(confusion_matrix(Y_validation, predictions))
print()

print("classification_report:")
print(classification_report(Y_validation, predictions))
print()



print('LinearDiscriminantAnalysis()')
mymod = LinearDiscriminantAnalysis()
mymod.fit(X_train, Y_train)
predictions = mymod.predict(X_validation)

print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print()

print("confusion_matrix:")
print(confusion_matrix(Y_validation, predictions))
print()

print("classification_report:")
print(classification_report(Y_validation, predictions))
print()


print('DecisionTreeClassifier()')
mymod = DecisionTreeClassifier()
mymod.fit(X_train, Y_train)
predictions = mymod.predict(X_validation)

print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print()

print("confusion_matrix:")
print(confusion_matrix(Y_validation, predictions))
print()

print("classification_report:")
print(classification_report(Y_validation, predictions))
print()



print('RandomForestClassifier()')
mymod = RandomForestClassifier()
mymod.fit(X_train, Y_train)
predictions = mymod.predict(X_validation)

print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print()

print("confusion_matrix:")
print(confusion_matrix(Y_validation, predictions))
print()

print("classification_report:")
print(classification_report(Y_validation, predictions))
print()



################################



''' FEATURE SELECTION !!!!!  RFE 

$$$ titanic logistic regression

http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-28.html

# https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/

Recursive Feature Elimination
The Recursive Feature Elimination (RFE) method is a feature selection approach. 
It works by recursively removing attributes and building a model on those attributes that remain.
 It uses the model accuracy to identify which attributes (and combination of attributes) 
 contribute the most to predicting the target attribute.

This recipe shows the use of RFE on the Iris flowers dataset to select 3 attributes
'''

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(dataset.data, dataset.target)
# summarize the selection of the attributes
print('Recursive Feature Elimination: Sepal Length is bad' )

print(rfe.support_)
print(rfe.ranking_)

# show order of features 
from sklearn.datasets import load_iris
# load data into PANDAS DATA FRAME 
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print(df.head(5))
print(' ')
print(df.describe())
#print(' ')



'''
# Feature Importance

Methods that use ensembles of decision trees (like Random Forest or Extra Trees) 
can also compute the relative importance of each attribute. These importance values can be used 
to inform a feature selection process.

This recipe shows the construction of an Extra Trees ensemble of the iris flowers dataset and 
the display of the relative feature importance.
'''

print('Model Feature Importance: petal length & petal width are super important - from last example, sepal width has weak feature power'   )

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)
# display the relative importance of each attribute
print(model.feature_importances_)


# STOP IRIS ###########################################


'''
PIMA Indian Diabetes PPPPPPPPPPPPPPPPPPPPPPPPPPPPP

FEATURE SELECTION

https://machinelearningmastery.com/feature-selection-machine-learning-python/

'''

# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_*100)
print('The scores suggest at the importance of plas,  mass, and age.')



# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)

print('\n You can see the scores for each attribute and the 4 attributes chosen \
   those with the highest scores): plas, test, mass and age.\n')
    
print(fit.scores_)
features = fit.transform(X)
# summarize selected features

print(features[0:5,:])





############ more feature selection below but do i need it? 


# Feature Extraction with RFE - this gave diff results ..... ? 
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print('You can see that RFE chose the the top 3 features as preg, mass and pedi.')


print(" ")

# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
print('Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into \
      a compressed form. \
Generally this is called a data reduction technique. A property of PCA is that you can \
 choose the number of dimensions or principal component in the transformed result. \
In the example below, we use PCA and select 3 principal components. \
You can see that the transformed dataset (3 principal components) bare little resemblance to the source data.')


print(" ")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# print(cancer.DESCR) # Print the data set description

print (cancer.keys())

# data = pd.DataFrame(cancer.data, columns=[cancer.feature_names])

df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns= np.append(cancer['feature_names'], ['target']))

print (df.keys())
print (df.shape)


# head
print(df.head(5))

# descriptions
print(df.describe())

# class distribution
print(df.groupby('target').size())

import numpy as geek
 
b = geek.zeros(2, dtype = int)
print("Matrix b : \n", b)
 
a = geek.zeros([2, 2], dtype = int)
print("\nMatrix a : \n", a)
 
c = geek.zeros([3, 3])
print("\nMatrix c : \n", c)



############################################
############################################

'''

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# NOT WORKING
'''

# https://www.fabienplisson.com/data-preprocessing/
# https://www.fabienplisson.com/cross-validation-classifiers/
# Feature importance
# https://www.fabienplisson.com/choosing-right-features/

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

X_train, X_train

rfc.fit(X_train, X_train)
importances = pd.DataFrame({'feature':data_x.columns,'importance':np.round(rfc.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
 
print(importances)
importances.plot.bar()





# https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/

import xgboost
# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()





sns.set(font_scale=1.4)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
sns.clustermap(data=corr, annot=True, fmt='d', cmap="Blues", annot_kws={"size": 16})

sns.set(font_scale=1.4)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
sns.heatmap(data=corr, annot=True, fmt='d', cmap="Blues", annot_kws={"size": 16})



###########################


plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()
#########################################################


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

d = dataset

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

########################################



# local file
print(os.getcwd())
os.chdir(os.path.dirname("C:\\Users\\User\\Desktop\\Coursera_Python_Wash\\"))
# os.chdir(os.path.dirname("C:\\Users\\petersid\\Documents\\Python\\"))
print(os.getcwd())
dataset2 = pd.read_csv('WA_Fn-UseC_-IT-Help-Desk.csv')


print(dataset2.shape)
print(' ')
# head
print(dataset2.head(5))
print(' ')
# descriptions
print(dataset2.describe())
print(' ')
# class distribution
print(dataset2.groupby('Requestor').size())
print(' ')

print(dataset2['Severity'].value_counts())

print(dataset2['Priority'].value_counts())

dataset2.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(7,7))
plt.show()


