#
# Todo：fix
# convLSTM
#

#
# Core Keras libraries
#
import numpy as np
from keras.models import Sequential,Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM, RepeatVector, TimeDistributed
from keras.layers import Bidirectional
from tensorflow.keras import layers, losses, optimizers, models
#
# For data conditioning
#
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

#
# Make results reproducible
#
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# 
# Other essential libraries
#
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_error
from numpy import array,stack

# Make our plot a bit formal
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

#
# Set input number of timestamps and training seconds
#
n_timestamp =10
pred_timestamp= 1
train_seconds = 2500  # number of seconds to train from
testing_seconds = 1040 # number of seconds to be predicted
n_epochs = 5
filter_on = 1


#
# Select model type
# 1: Single cell
# 2: Stacked
# 3: conv LSTM
#
model_type = 3

#
# Data preprocessing
#
url = "data/result_5000.json"
dataset = pd.read_json(url,typ='series')

print(dataset.values.shape)
# print(dataset.values.size)
row = len(array(dataset.values[0]))
col = len(array(dataset.values[0])[0])

for i in range(dataset.values.size):
# for i in range(4):
    # Numpy 中的 ravel() 和 flatten()两个函数可以对多维数据进行扁平化操作。
    # flatten() 返回的是一个数组的的副本，新的对象；ravel() 返回的是一个数组的非副本视图。
    dataset.values[i] = array(dataset.values[i])
    # print(dataset.values[i])

# if filter_on == 1:                          # 用于激活数据过滤器
#     # dataset.values = medfilt(dataset.values.all, 3)
#     dataset.values = gaussian_filter1d(dataset.values.all, 1.2)

dataset=stack([x for x in dataset])         #将serise 转为 三维矩阵
print(dataset.shape)

# Add a channel dimension since the images are grayscale.
dataset = np.expand_dims(dataset, axis=-1)

#
# Set number of training and testing data
#
train_set = dataset[0:train_seconds]
test_set = dataset[train_seconds: train_seconds+testing_seconds]
training_set = train_set
testing_set = test_set


# #
# # Normalize data first
# #
# sc = MinMaxScaler(feature_range = (0, 1))       # 将数据标准化，范围是0到1
# training_set_scaled = sc.fit_transform(training_set)
# testing_set_scaled = sc.fit_transform(testing_set)

#
# Split data into n_timestamp
#设置观察多久、预测多久
#
def data_split(sequence, n_timestamp, pred_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp + pred_timestamp
        if end_ix > len(sequence)-1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x = sequence[i:end_ix-pred_timestamp]
        seq_y = sequence[end_ix-pred_timestamp:end_ix]
        # seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

X_train, y_train = data_split(training_set, n_timestamp,pred_timestamp)
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],row*col)
X_test, y_test = data_split(testing_set, n_timestamp, pred_timestamp)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], row*col)
print(X_train.shape)
print(X_train.shape[0], X_train.shape[1])

#
# # 载入模型
# model = load_model('model/model_150.h5')
#
# # 评估模型
# loss,accuracy = model.evaluate(X_test,y_test)


if model_type == 1:
    # Single cell LSTM
    model = Sequential()
    model.add(LSTM(units = 50, activation='relu',input_shape = (X_train.shape[1], row*col)))
    # model.add(Dense(row*col))
    model.add(RepeatVector(pred_timestamp))
    model.add(TimeDistributed(Dense(row*col)))
if model_type == 2:
    # Stacked LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], row*col)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(row*col))
if model_type == 3:
    # Conv LSTM
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *X_train.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model =  models.Model(inp, x)
    model.compile(
        loss=losses.mean_squared_error, optimizer=optimizers.Adam(),
    )



model.summary()


#
# Start training
#
# model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = 32)
# loss,accuracy = model.evaluate(X_test,y_test)


loss = history.history['loss']
epochs = range(len(loss))

# model.save("model/model_%d.h5"%n_epochs)

#
# Get predicted data
#
y_predicted = model.predict(X_test)
# y_predicted = y_predicted.astype(int)
y_predicted = np.round(y_predicted).astype(int)
print(y_predicted.shape)
print("Predict:")
print(y_predicted[0][0].reshape(row,col))
print("True:")
print(y_test[0][0].reshape(row,col))

#
# 'De-normalize' the data
# #
# y_predicted_descaled = sc.inverse_transform(y_predicted)
# y_train_descaled = sc.inverse_transform(y_train)
# y_test_descaled = sc.inverse_transform(y_test)
# y_pred = y_predicted.ravel()
# y_pred = [round(yx, 2) for yx in y_pred]
# y_tested = y_test.ravel()


#
# Show results
#

plt.plot(epochs, loss, color='black')
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.title("Training curve")


mse = mean_squared_error(y_test.reshape((y_test.shape[0],-1)), y_predicted.reshape(y_test.shape[0],-1))
# r2 = r2_score(y_test.reshape((y_test.shape[0],-1)), y_predicted.reshape(y_test.shape[0],-1))
mae= mean_absolute_error(y_test.reshape((y_test.shape[0],-1)), y_predicted.reshape(y_test.shape[0],-1))
print("mse=" + str(round(mse,2)))
# print("r2=" + str(round(r2,2)))
print("mae=" + str(round(mae,2)))
plt.show()