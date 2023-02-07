# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 22:55:39 2022

@author: user


https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

Need: Keras and a backend (Theano or TensorFlow) installed and configured

Load Data
Define Keras Model
Compile Keras Model
Fit Keras Model
Evaluate Keras Model
Tie It All Together
Make Predictions

How to deploy with GPU on AWS
https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/

Make predictions with Keras
https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/

Deep learning
optical recogniton examples
https://www.edureka.co/blog/deep-learning-with-python/

"""


# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os
print(' ')

# local file
print(os.getcwd())
os.chdir(os.path.dirname("C:\\Users\\user\\Desktop\\DeepLearning\\"))
print(os.getcwd())

...
# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]


# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# for jupyter
# fit the keras model on the dataset without progress bars
# model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make probability predictions with the model
predictions = model.predict(X)
# round predictions 
# rounded = [round(x[0]) for x in predictions]


# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))




