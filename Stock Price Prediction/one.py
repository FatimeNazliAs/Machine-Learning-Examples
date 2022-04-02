# description: this program uses artificial neural network called long short term memory
# to predict closing stock price of a corporation (Apple Inc.) using past 60 day stock price
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
print(df)

plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=15)
plt.ylabel('Close Price USD', fontsize=15)
plt.show()

# new data frame with 'close' column
data = df.filter(['Close'])
# convert dataframe to numpy array
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)  # %80 training set
# training_data_len

# To decrease the computational cost of the data in the table, we will scale the stock values to values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# create training data set
train_data = scaled_data[0:training_data_len, :]
print(len(train_data))
# print(train_data)
# split data into x_train and y_train data sets

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])  # past 60 values
    y_train.append(train_data[i, 0])  # first 60 values
    if i <= 60:
        print(x_train)
        print(y_train)
        print("\n")

# As the LSTM needs that the data to be provided in the 3D form, we first transform the training and test data to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compile model
# Adam optimization is a stochastic gradient descent method that is based on
# adaptive estimation of first-order and second-order moments.
model.compile(optimizer='adam', loss='mean_squared_error')
# train model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create testing data set
# create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
# LSTM expects 3dimensional array
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)

train = data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

plt.show()

print(valid)

apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
new_df = apple_quote.filter(['Close'])
# get last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# scale data 0-1
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])