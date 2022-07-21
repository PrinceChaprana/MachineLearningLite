from pickletools import optimize
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pkg_resources import add_activation_listener
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#read Data
data = pd.read_csv(r"A_Z Handwritten Data.csv").astype('float32')

#split data into image and labels
x = data.drop('0',axis = 1)
y = data['0']

#Reshaping the data so can be displayed as an image

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.2)

train_x = np.reshape(train_x.values,(train_x.shape[0],28,28))
test_x = np.reshape(test_x.values,(test_x.shape[0],28,28))

#converted the 784 coloums of pixel to 28 28 pixel image above

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',}
#ALL Values are in floating value we create dict to map the integer with character

#ploting number of alphabets in dataset

y_int = np.int0(y)
count = np.zeros(26,dtype='int')
for i in y_int:
    count[i] += 1
alphabets = []
for i in word_dict.values():
    alphabets.append(i)

fig,ax = plt.subplots(1,1,figsize = (10,10))
ax.barh(alphabets,count)

plt.xlabel("number of elements")
plt.ylabel("alphabets")
#plt.grid()
#plt.show()

#shuffling the data
shuff = shuffle(train_x[:100])
fig,ax = plt.subplots(3,3,figsize=(10,10))
axes = ax.flatten()

for i in range(9):
    _,shu = cv2.threshold(shuff[i],30,200,cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i],(28,28)),cmap="Greys")
#plt.show()

#reshaping data for the training and testing purpose

train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)

test_X = test_x.reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2],1)

#now they are ready to put in model

train_yOHE = to_categorical(train_y,num_classes = 26,dtype = 'int')
test_yOhe = to_categorical(test_y,num_classes = 26,dtype = 'int')

'''
The convolution layers are generally followed by maxpool layers that are used to reduce the number of features extracted and ultimately the output of the maxpool and layers and convolution layers are flattened into a vector of single dimension and are given as an input to the Dense layer (The fully connected network).'''

model = Sequential()

model.add(Conv2D(filters = 32,kernel_size=(3,3),activation = 'relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size = (2,2),strides = 2))

model.add(Conv2D(filters = 64,kernel_size=(3,3),activation = 'relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size = (2,2),strides = 2))

model.add(Conv2D(filters = 128,kernel_size=(3,3),activation = 'relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size = (2,2),strides = 2))

model.add(Flatten())

model.add(Dense(64,activation = "relu"))
model.add(Dense(128,activation = "relu"))

model.add(Dense(26,activation = 'softmax'))


#compiling model

#model.compile(optimizer = Adam(learning_rate = 0.001),loss = 'categorical_crossentropy' ,metrics = ['accuracy'] )

#history = model.fit(train_X,train_yOHE,epochs = 1,validation_data = (test_X,test_yOhe))

#model.summery()
#model.save(r"model_alpha.h5")

#doing some prediction
fig,axes = plt.subplots(3,3,figsize = (8,9))
axes = axes.flatten()

for i,ax in enumerate(axes):
    img = np.reshape(test_X[i],(28,28))
    ax.imshow(img,cmap="Greys")

    pred = word_dict[np.argmax(test_yOhe[i])]
    ax.set_title("Prediction"+pred)
    ax.grid()
plt.show()