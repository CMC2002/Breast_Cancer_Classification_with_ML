#!/usr/bin/env python
# coding: utf-8

# In[143]:


import os
import glob
import scipy.io as sio
import numpy as np

## Read data
## 抓出benign中要用的data
## save to x_list_be && y_list_be
path_b = r"data/benign/*.mat"
x_list_be=[]
y_list_be=[]
x_list=[]
y_list=[]
for i in glob.glob(path_b):
    data_b = sio.loadmat(i)
    x_data_b = data_b.get('TDM_VOI')
    y_data_b = data_b.get('Type_lable')
    x_data_be = x_data_b.tolist()
    y_data_be = y_data_b.tolist()
    x_list_be.append(x_data_be)
    y_list_be.append(y_data_be)

## 抓出malignancy中要用的data
## save to x_list_ma && y_list_ma
path_m = r"data/malignancy/*.mat"
x_list_ma=[]
y_list_ma=[]
for i in glob.glob(path_m):
    data_m = sio.loadmat(i)
    x_data_m = data_m.get('TDM_VOI')
    y_data_m = data_m.get('Type_lable')
    x_data_ma = x_data_m.tolist()
    y_data_ma = y_data_m.tolist()
    x_list_ma.append(x_data_ma)
    y_list_ma.append(y_data_ma)
    
x_list = x_list_be + x_list_ma
y_list = y_list_be + y_list_ma


# In[144]:


import random

x_test_be=[]
x_test_ma=[]
y_train_be=[]
y_train_ma=[]
y_test_be=[]
y_test_ma=[]
## 隨機選取各40個data
## x_list_be && x_list_ma
## x_train：需要train的80筆資料
x_train_be = random.sample(x_list_be,40)
x_train_ma = random.sample(x_list_ma,40)
x_train = x_train_be + x_train_ma
for i in x_train:
    if i in x_train_be:
        y_train_be.append(0)
    else:
        y_train_ma.append(1)
y_train =  y_train_be +  y_train_ma

## 將為選到的10筆data放置為test data
## x_test_be && x_test_ma
## x_test：需要test的20筆資料
#x_test_be = list(set(x_list_be).difference(set(x_train_be)))
for i in x_list_be:
    if i not in x_train_be:
        x_test_be.append(i)
        y_test_be.append(0)
for j in x_list_ma:
    if j not in x_train_ma:
        x_test_ma.append(j)
        y_test_ma.append(1)
x_test = x_test_be + x_test_ma
y_test = y_test_be + y_test_ma

print(len(y_test))
print(y_test)


# In[145]:


## 全部的x data： x_list ## 全部的y data： y_list
## train data40：x_train、y_train
## test data10：x_test、y_test
from keras.utils import np_utils
import numpy as np

x_train = np.array(x_train)
x_test = np.array(x_test)
#print(type(x_train))
## 給他一個通道來放後面要存的東西，x_train.shape[0]=80(數量)
x_train80 = x_train.reshape(x_train.shape[0],64,64,64,1).astype('float32')
x_test20 = x_test.reshape(x_test.shape[0],64,64,64,1).astype('float32')
## Standardize
x_train80_norm = x_train80 / 255
x_test20_norm = x_test20 / 255
## Label oneHot-encoding
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

#print(type(y_train))


# In[146]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv3D,MaxPooling3D
from keras.layers.normalization import BatchNormalization

## 總之先開始建吧
model = Sequential()

## Create CN Layer 1
model.add(Conv3D(filters=8,
                 kernel_size=(9,9,9),
                 padding='same',
                 input_shape=(64,64,64,1),
                 activation='sigmoid'))

##Add Dropout Layer
model.add(Dropout(0.25))

##Create Max-Pooling 1
model.add(MaxPooling3D(pool_size=(2,2,2)))

## Create CN Layer 2
model.add(Conv3D(filters=16,
                 kernel_size=(9,9,9),
                 padding='same',
                 input_shape=(64,64,64,1),
                 activation='sigmoid'))
##Add Dropout Layer
model.add(Dropout(0.25))

##Create Max-Pooling 2
model.add(MaxPooling3D(pool_size=(2,2,2)))

## Create CN Layer 3
model.add(Conv3D(filters=32,
                 kernel_size=(9,9,9),
                 padding='same',
                 input_shape=(64,64,64,1),
                 activation='sigmoid'))

##Add Dropout Layer
model.add(Dropout(0.25))

##Create Max-Pooling 3
model.add(MaxPooling3D(pool_size=(2,2,2)))

## Create CN Layer 4
model.add(Conv3D(filters=128,
                 kernel_size=(9,9,9),
                 padding='same',
                 input_shape=(64,64,64,1),
                 activation='sigmoid'))

##Add Dropout Layer
model.add(Dropout(0.25))

##Create Max-Pooling 4
model.add(MaxPooling3D(pool_size=(2,2,2)))

## Create CN Layer 5
model.add(Conv3D(filters=256,
                 kernel_size=(9,9,9),
                 padding='same',
                 input_shape=(64,64,64,1),
                 activation='sigmoid'))

##Add Dropout Layer
model.add(Dropout(0.25))

##Create Max-Pooling 5
model.add(MaxPooling3D(pool_size=(2,2,2)))


# In[147]:


## conv跟maxpool都用完了
## 就來建立神經網路(Flatten)

model.add(Flatten())
#model.add(BatchNormalization())
model.add(Dense(128,activation='sigmoid'))
model.add(Dropout(0.25))

##建立輸出層 我們輸出是2個output
model.add(Dense(2,activation='softmax'))


# In[148]:


## 查看模型摘要
model.summary()
print("")


# In[149]:


### 定義訓練 && 進行訓練
##定義訓練
#categorical_crossentropy
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

##訓練開始
##x_train80_norm
##y_train_onehot
train_history = model.fit(x = x_train80_norm,
                         y = y_train_onehot,validation_split=0.2,
                         epochs=10,batch_size=10,verbose=2)


# In[150]:


import os

def isDisplayAvl():
    return 'DISPLAY' in os.environ.keys()

import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()
    
def plot_images_labels_predict(images, labels, prediction, idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25 : num = 25
    for i in range(0,num):
        ax=pkt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title = "l=" + str(labels[idx])
        if len(prediction) > 0:
            title = "l={},p={}".format(str(labels[idx]),str(prediction[idx]))
        else:
            tilte = "l={}".format(str(labels[idx]))
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
    
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


# In[151]:


from keras.utils import *
#if isDisplayAvl():
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')


# In[152]:


scores = model.evaluate(x_test20_norm,y_test_onehot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

