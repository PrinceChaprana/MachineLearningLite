#this uses LSTM neural model
#Dataset is of Tata Global Beverages Stock Data
#

from pickletools import optimize
from pyexpat import features
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler

#dataset Read
df = pd.read_csv("Tata.csv")

#Analyzing the Closing price from DataFrame
df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index = df["Date"]

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')

#sort the dataset on date time and filter them

data = df.sort_index(ascending=True,axis=0 )
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]


#Normalize the new Filterd Dataset
scaler = MinMaxScaler(feature_range=(0,1))
final_data = new_data.values

train_data = final_data[0:987,:]
valid_data = final_data[987:,:]

new_data.index = new_data.Date
new_data.drop("Date",axis=1,inplace=True)
scaler = MinMaxScaler(feature_range=(0,1))
#scaled_Data = scaler.fit_transform(final_data)
print(final_data )
final_data = new_data.values
scaled_Data = scaler.fit_transform(final_data)
x_train,y_train = [],[]

for i in range(60,len(train_data)):
    x_train.append(scaled_Data[i-60:i,0])
    y_train.append(scaled_Data[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Building the LSTM Model and Training it
lstm_model = Sequential()
lstm_model.add(LSTM(units=50,return_sequences = True,input_shape=(x_train.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data = new_data[len(new_data)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

lstm_model.compile(loss = 'mean_squared_error',optimizer = 'adam')
lstm_model.fit(x_train,y_train,epochs=1,batch_size = 1,verbose = 2)

#take the sample of dataset to make prediction
X_test = []
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predict_close_price = lstm_model.predict(X_test)
predict_close_price = scaler.inverse_transform(predict_close_price)

#SAVINGT THE MODEL
lstm_model.save("predict_model.h5")

#visualize train data
train_data = new_data[:987]
valid_data = new_data[987:]
valid_data['Predictions'] = predict_close_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
plt.show()