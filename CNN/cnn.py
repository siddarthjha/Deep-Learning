"""

This is a 5 layers Sequential Convolutional Neural Network for digits recognition trained on MNIST dataset.
Firstly, I will prepare the data (handwritten digits images) then i will focus on the CNN modeling and evaluation.

It follows three main parts:

The data preparation
The CNN modeling and evaluation
The results prediction and submission

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
sns.set(style='white', context='notebook', palette='deep')
# Using tensorflow backend

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# You can take your own data or else download from kaggle
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

g = sns.countplot(Y_train)

Y_train.value_counts()
# Check for null and missing values
X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Set the random seed
