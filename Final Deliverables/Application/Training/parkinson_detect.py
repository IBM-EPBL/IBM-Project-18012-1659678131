# -*- coding: utf-8 -*-
"""parkinson_detect.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NRvxGOWr5K1rLKsgF_SU4PLzvDXl4i8J

#Data Collection 
###Download the dataset
"""

from google.colab import drive
drive.mount('/content/drive')

import warnings
warnings.filterwarnings("ignore")

"""#Image Pre-processing
Importing the necessary libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile as zf
import os
import random
import cv2
import pickle
from imutils import build_montages
from imutils import paths
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from skimage import feature
from google.colab.patches import cv2_imshow

sns.set()
os.getcwd()

"""##Functions to load and quantify the images"""

def quantify_image(image):
    features = feature.hog(image, 
                           orientations=9, 
                           pixels_per_cell=(10,10), 
                           cells_per_block=(2,2), 
                           transform_sqrt=True, 
                           block_norm="L1")
    return features

def load_split(path):
    path_images = list(paths.list_images(path))
    data=[]
    labels=[]

    for path_image in path_images:
        label = path_image.split(os.path.sep)[-2]
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200,200))
        image = cv2.threshold(image,0,225,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_image(image)
        data.append(features)
        labels.append(label)

    return (np.array(data), np.array(labels))

"""#Using spiral images
##Defining the path for training data and testing data
"""

handle_spiral = zf.ZipFile('/content/drive/MyDrive/dataset.zip')
handle_spiral.extractall('dataset')
handle_spiral.close()

s_w_train_healthy = os.listdir('dataset/spiral_wave/training/healthy/')
s_w_train_park = os.listdir('dataset/spiral_wave/training/parkinson/')

fp_s_w_train_healthy = 'dataset/spiral_wave/training/healthy/'
fp_s_w_train_park = 'dataset/spiral_wave/training/parkinson/'

s_w_test_healthy = os.listdir('dataset/spiral_wave/testing/healthy/')
s_w_test_park = os.listdir('dataset/spiral_wave/testing/parkinson/')

fp_s_w_test_healthy = 'dataset/spiral_wave/testing/healthy/'
fp_s_w_test_park = 'dataset/spiral_wave/testing/parkinson/'

Xtrain = []
Xtest = []
outputs = []
Ytrain = []
Ytest= []

for i in s_w_train_healthy:
  image = cv2.imread(fp_s_w_train_healthy+i)
  image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image , (200,200))
  image =cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify_image(image)
  Xtrain.append(features)
  Ytrain.append('healthy')

for i in s_w_train_park:
  image = cv2.imread(fp_s_w_train_park+i)
  image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image , (200,200))
  image = cv2.threshold(image ,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify_image(image)
  Xtrain.append(features)
  Ytrain.append('parkinson')

for i in s_w_test_healthy:
  image = cv2.imread(fp_s_w_test_healthy+i)
  outputs.append(image)
  image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image , (200,200))
  image = cv2.threshold(image ,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify_image(image)
  Xtest.append(features)
  Ytest.append('healthy')

for i in s_w_test_park:
  image = cv2.imread(fp_s_w_test_park+i)
  outputs.append(image)
  image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image , (200,200))
  image = cv2.threshold(image ,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify_image(image)
  Xtest.append(features)
  Ytest.append('parkinson')

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)
Xtrain

Ytrain

Xtest

Ytest

"""##Label Encoding"""

label_encoder = LabelEncoder()
Ytrain = label_encoder.fit_transform(Ytrain)
Ytest = label_encoder.transform(Ytest)
print(Xtrain.shape, Ytrain.shape)

Ytrain

Ytest

"""#MODEL BUILDING
###Training the Model
"""

print("Training model....")
model = RandomForestClassifier(n_estimators=100)
model.fit(Xtrain,Ytrain)

preds = model.predict(Xtest)
preds

"""##Model Evalution"""

cnf = confusion_matrix(Ytest,preds)
cnf

plt.figure(figsize=(5,5))
sns.heatmap(cnf , annot=True , cmap="coolwarm" , cbar=False)
plt.show()

acc = metrics.accuracy_score(Ytest,preds)
acc

indexes = np.random.randint(0,30,25)
indexes

"""#Testing the Model"""

testpath=list(paths.list_images(fp_s_w_train_healthy))
idxs=np.arange(0,len(testpath))
idxs=np.random.choice(idxs,size=(25,),replace=False)
images=[]

for i in idxs:
    image=cv2.imread(testpath[i])
    output=image.copy()
    output=cv2.resize(output,(128,128))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image,(200,200))
    image=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    features= quantify_image(image)
    preds=model.predict([features])
    label=label_encoder.inverse_transform(preds)[0]
    color=(0, 255, 0) if label=="healthy" else (0, 0, 255)
    cv2.putText(output,label, (3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    images.append(output)

'''montage = build_montages(images,(128,128),(5,5))[0]
cv2.imshow(montage)
cv2.waitKey(0)'''

montage=build_montages(images,(128, 128),(5, 5))[0]
cv2_imshow(montage)
cv2.waitKey(0)

"""##Predicting the model-Accuracy and Confusion Matrix"""

predictions = model.predict(Xtest)

cm = confusion_matrix(Ytest, predictions).flatten()
print(cm)
(tn, fp, fn, tp) = cm
accuracy = (tp + tn) / float(cm.sum())
print(accuracy)

"""#Save the model"""

pickle.dump(model,open('parkinson.pkl','wb'))