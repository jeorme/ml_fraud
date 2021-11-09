# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 13:01:51 2021

@author: sgonzalez
"""
import math as math
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
#from sklearn.preprocessing import StandardScaler # data normalization
from tensorflow.keras import layers
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout

import pandas as pd
import pennylane as qml
import numpy as np
from sklearn.model_selection import train_test_split # data split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

#%% LOAD DATA + STATISTICS
"Loading the data and examining it"

df = pd.read_csv('creditcard.csv')

#drop time variable (not relevant) and duplicated data
df.drop('Time', axis = 1, inplace = True)
print('Number of duplicated data arrays', df.duplicated().sum())

df.drop_duplicates(keep=False,inplace=True)

#check data load with a print
#print(df.head()) 

cases = len(df)
nonfraud_count = len(df[df.Class == 0])
fraud_count = len(df[df.Class == 1])
fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

print('Parameters of the data')
print('--------------------------------------------')
print('Total number of cases are', cases)
print('Number of Non-fraud cases are', nonfraud_count)
print('Number of Non-fraud cases are',fraud_count)
print('Percentage of fraud cases is' ,fraud_percentage , '%')
print('--------------------------------------------')

"NB: fraud/nonfraud ratio is only 0.17%"

nonfraud_cases = df[df.Class == 0]
fraud_cases = df[df.Class == 1]

"""
print("Data statistics")
print('--------------------------------------------')
print('Non-fraud stats')
print(nonfraud_cases.Amount.describe())
print('--------------------------------------------')
print('Fraud stats')
print(fraud_cases.Amount.describe())
print('--------------------------------------------')
"""
#%% DATA SPLIT
"DATA SPLIT"

Vs = df.drop('Class', axis = 1).values #Vs vsalues
y = df['Class'].values #Fraud/non-fraud array

normalise = False

if normalise:
    #print('First few values BEFORE normalisation') 
    # print(Vs[0:5,-1])
    # plt.plot(Vs[0:200,-1])
    #plt.show()
    Vs[:,-1] = Vs[:,-1]/np.linalg.norm(Vs[:,-1])
    #print('First few values AFTER normalisation')
    # print(Vs[0:5,-1])
    # plt.plot(Vs[0:200,-1])
    #plt.show()

#truncate data for testing
Vs = Vs[90000:100000]
y = y[90000:100000]

testtotrain = 0.2
X_train, X_test, y_train, y_test = train_test_split(Vs, y, test_size = testtotrain, random_state = 0)

#crop for testing
'''
def one_hot(labels):     
    depth = 16                       # 10 classes + 6 zeros for padding
    indices = labels.astype(np.int32)    
    one_hot_labels = np.eye(depth)[indices].astype(np.float32) 
    return one_hot_labels
'''
#%% QUANTUM LAYER
"""We make a quantum layer using Pennylane. This will then be converted into a 
classical Keras layer using the Pennylane class pennylane.qnn.KerasLayer. For more
information visit: https://pennylane.ai/qml/demos/tutorial_qnn_module_torch.html"""

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

"""For the QNode to be successfully converted to a layer in Keras, we need to 
provide the details of the shape of each trainable weight for them to be initialized. 
The ``weight_shapes`` dictionary maps from the argument names of the QNode to corresponding shapes"""
n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
print("Quantum layer created")
#%% CLASSICAL LAYERS + INTEGRATE
# Create CLASSICAL layers

tf.keras.backend.set_floatx('float64')

model=Sequential() #create model
print("model created")
#add layers to our hybrid model
model.add(Dense(units=2,kernel_initializer='he_normal',activation='relu'))
model.add(qlayer)
model.add(Dense(units=2,kernel_initializer='he_normal',activation='softmax'))

print("Quantum and classical layers added")
model.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = len(X_train), epochs = 5,validation_split=0.2)

#%%EVALUATION 

# Plot history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# Plot history of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

print('Max val_acc achieved: %.2f' %(max(history.history['val_accuracy'])*100), '%')
print('Max acc achieved: %.2f' %(max(history.history['accuracy'])*100), '%')

#Confusion matrix
y_pred = model.predict(X_test)
ann_cm = confusion_matrix(y_test, y_pred)
ann_acc = round(accuracy_score(y_pred,y_test) * 100, 2)
print(ann_cm)
print(ann_acc,'%')
print(classification_report(y_pred,y_test))