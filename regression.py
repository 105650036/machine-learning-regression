# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:51:34 2019

@author: Administrator
"""
import pandas as pd
import keras as keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('F:/class/machine learning/train-input.csv', delim_whitespace=True, header=None,delimiter=",")
df2 = pd.read_csv('F:/class/machine learning/train-output.csv', delim_whitespace=True, header=None)
dataset = np.loadtxt("F:/class/machine learning/train-input.csv", delimiter=",")
test1 = np.loadtxt("F:/class/machine learning/valid-input.csv", delimiter=",")
output=np.loadtxt("F:/class/machine learning/testinput.csv", delimiter=",")

x_test=test1[:,:]
X = dataset[:,:]
Xoutput=output[:,:]
mean=X.mean(axis=0)
X-=mean
Xoutput-=mean
std=X.std(axis=0)
X/=std
Xoutput/=std
Xtrain=X[:,:]
Xtest=X[10000:12967,:]
print(X[25])
Y = df2.values[:,:]
print(Y)



x1=[[10,12,20],[12,15,18],[22,15,19],[87,20,55],[49,13,27],[99,22,33],[16,17,21]]
x1=np.array(x1)
print(x1)
model = Sequential()
model.add(Dense(units=21,
                input_dim=17,
                kernel_initializer='normal',
                activation='relu'
                ))
model.add(Dense(units=100,
                kernel_initializer='normal',
                activation='relu'
                ))
model.add(Dense(units=1,
                kernel_initializer='normal',
                activation='relu'
                ))
SGD=keras.optimizers.SGD(lr=0.15, momentum=0, decay=0, nesterov=False)
model.compile(loss='mean_absolute_error', optimizer='Adadelta',metrics=['accuracy'])

history=model.fit(Xtrain, Y, batch_size=256, epochs=1000, initial_epoch=0)

y_pred = model.predict(Xoutput)
# test the model
score = model.evaluate(Xtrain, Y, batch_size=256)
print (score)
print(y_pred)
model.save('my_model.h5')
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()