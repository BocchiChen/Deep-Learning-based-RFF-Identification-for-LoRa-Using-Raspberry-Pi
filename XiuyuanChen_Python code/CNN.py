# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Activation
from keras.models import Sequential
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
import random
from keras.utils import np_utils
import matplotlib.pyplot as plt
import scipy.signal

data = []
x_train = []
x_valid = []
x_test = []
random_x_train = []
random_x_test = []
random_y_train = []
random_y_test = []
sample_length = 1100
train_set_length = 900
valid_set_length = 100
mid = sample_length - valid_set_length
train_interval = 900
valid_interval = 100
test_interval = 100
device_number = 5
valid_set_length = 500

# Read the received LoRa packets
def loadDataSet(index):
    sample_num = random.sample(range(0, sample_length), sample_length)
    data = []
    file_path = './npydata/Sxx_auto/num' + index + '.npy'
    loadData = np.load(file_path)
    data.append(loadData[0:sample_length][:][:])
    xp = np.array(data).reshape(sample_length,256,63)
    for x in range(0,train_set_length):
        x_train.append(xp[sample_num[x]][:][:])
    for y in range(train_set_length,mid):
        x_valid.append(xp[sample_num[y]][:][:]) 
    for z in range(mid,sample_length):
        x_test.append(xp[sample_num[z]][:][:]) 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set up the labels for each sample
def setLabels():
    global y_train
    for i in range(len(y_train)):
        if i < train_interval:
            y_train[i] = 0
        elif i >= train_interval and i < train_interval*2:
            y_train[i] = 1
        elif i >= train_interval*2 and i < train_interval*3:
            y_train[i] = 2
        elif i >= train_interval*3 and i < train_interval*4:
            y_train[i] = 3
        elif i >= train_interval*4 and i < train_interval*5:
            y_train[i] = 4
    y_train = y_train.reshape(-1,1)
    
    global y_valid
    for k in range(len(y_valid)):
        if i < valid_interval:
            y_valid[i] = 0
        elif i >= valid_interval and i < valid_interval*2:
            y_valid[i] = 1
        elif i >= valid_interval*2 and i < valid_interval*3:
            y_valid[i] = 2
        elif i >= valid_interval*3 and i < valid_interval*4:
            y_valid[i] = 3
        elif i >= valid_interval*4 and i < valid_interval*5:
            y_valid[i] = 4
    y_valid = y_valid.reshape(-1,1)
    
    global y_test
    for j in range(len(y_test)):
        if j < test_interval:
            y_test[j] = 0
        elif j >= test_interval and j < test_interval*2:
            y_test[j] = 1
        elif j >= test_interval*2 and j < test_interval*3:
            y_test[j] = 2
        elif j >= test_interval*3 and j < test_interval*4:
            y_test[j] = 3
        elif j >= test_interval*4 and j < test_interval*5:
            y_test[j] = 4
    y_test = y_test.reshape(-1,1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Disrupt the order of test and training sets
def disruptSets():
    global random_x_train, random_y_train, random_x_test, random_y_test, random_y_train, y_valid
    
    # Generate random index
    L1 = random.sample(range(0, device_number*train_interval), device_number*train_interval)
    L2 = random.sample(range(0, device_number*test_interval), device_number*test_interval)
    
    for x in range(len(L1)):
        random_x_train.append(x_train[L1[x]][:][:])
        random_y_train.append(y_train[L1[x]][:])
    
    random_x_train = np.array(random_x_train).reshape(device_number*train_interval,256,63,1)
    random_y_train = np.array(random_y_train)
    
    for y in range(len(L2)):
        random_x_test.append(x_test[L2[y]][:][:])
        random_y_test.append(y_test[L2[y]][:])
    
    random_x_test = np.array(random_x_test).reshape(device_number*test_interval,256,63,1)
    random_y_test = np.array(random_y_test)
    
    random_y_train = np_utils.to_categorical(random_y_train,device_number)
    random_y_test = np_utils.to_categorical(random_y_test,device_number)
    y_valid = np_utils.to_categorical(y_valid,device_number)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def CNNModel():
    inputShape = (256,63,1)
    
    '''
    model.add(GRU(512,input_shape = inputShape,return_sequences=True))
    model.add(GRU(512,return_sequences=True))
    
    # Build MLP architecture
    model.add(Flatten(input_shape = inputShape))
    model.add(Dense(1024,kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    model.add(Dense(1024,kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    model.add(Dense(1024,kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    '''

    # Build CNN architecture
    model = Sequential()
    model.add(Conv2D(8,(3, 3),input_shape = inputShape,padding='same',kernel_regularizer=keras.regularizers.l2(0.0001))) # Convolution layer
    model.add(BatchNormalization(axis=-1)) # Batch Normalization layer
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(16,(3, 3),padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(32,(3, 3),padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dropout(0.5)) # Dropout layer
    model.add(Dense(128,kernel_regularizer=keras.regularizers.l2(0.0001))) # Fully connected layer
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    model.summary()
    
    # Compile the NN model
    model.compile(loss= 'categorical_crossentropy',metrics=['accuracy'],optimizer= optimizers.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.3, patience=10, mode='auto')
    history = model.fit(random_x_train,random_y_train, epochs = 60, batch_size=32, validation_data=(x_valid, y_valid),callbacks=[reduce_lr])
    
    # Print the test results of the model
    score = model.evaluate(random_x_test, random_y_test)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    
    # Depict validation and trainning performance figures
    plt.subplot(121)
    plt.plot(scipy.signal.savgol_filter(history.history['loss'],43,2))
    plt.plot(scipy.signal.savgol_filter(history.history['val_loss'],43,2))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='lower left')
    plt.subplot(122)
    plt.plot(scipy.signal.savgol_filter(history.history['accuracy'],43,2))
    plt.plot(scipy.signal.savgol_filter(history.history['val_accuracy'],43,2))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='lower right')
    plt.show()
    
    # Save model and the weights
    model.save('rffi.h5')
    model.save_weights('weights.h5')

# Main function
if __name__ == '__main__':
    for number in range (1,device_number+1):
        loadDataSet(str(number))
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_valid = np.array(x_valid)
    
    y_train = np.zeros(device_number*train_interval)
    y_test = np.zeros(device_number*test_interval)
    y_valid = np.zeros(device_number*valid_interval)
    
    setLabels()
    x_train = x_train.reshape(device_number*train_interval,256,63)
    x_valid = x_valid.reshape(device_number*valid_interval,256,63)
    x_valid = np.array(x_valid).reshape(-1,256,63,1)
    x_test = x_test.reshape(device_number*test_interval,256,63)
    
    disruptSets()
    
    CNNModel()

