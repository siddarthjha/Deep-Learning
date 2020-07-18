import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
s = tf.InteractiveSession()

# Importing the Dataset and splitting into Train and Test

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
	
	# normalize x
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255
	
    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
       	X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([0], -1)
    
		
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

## Printing dimensions
print(X_train.shape, y_train.shape)

## Visualizing the first digit
plt.imshow(X_train[0], cmap="Greys")
plt.show()

## Changing dimension of input images from N*28*28 to  N*784
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
print('Train dimension:')
print(X_train.shape)
print('Test dimension:')
print(X_test.shape)

