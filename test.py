
#
# Make results reproducible
#
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
# tensorflow v2
import tensorflow
tensorflow.random.set_seed(2)

#
# Other essential libraries
#
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
from numpy import array,stack

# Make our plot a bit formal
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

# ------------------修改此处----------------------------------
#
# Set parameters
#
grid_size = 4000
n_timestamp = 120
pred_timestamp= 60

train_seconds = 2500  # number of seconds to train from
testing_seconds = 1040 # number of seconds to be predicted
n_epochs = 25
# -----------------END-----------------------------------------


#
# Data preprocessing
#
url = "data/result_%d.json"%grid_size
dataset = pd.read_json(url,typ='series')

row = len(array(dataset.values[0]))
col = len(array(dataset.values[0])[0])

for i in range(dataset.values.size):
    dataset.values[i] = array(dataset.values[i]).reshape(1,-1) # 矩阵降为一维
    # print(dataset.values[i])

dataset=stack([x for x in dataset])         # 将serise 转为 三维矩阵
print("dataset shape",dataset.shape)

#
# Set number of training and testing data
#
train_set = dataset[0:train_seconds]
test_set = dataset[train_seconds: train_seconds+testing_seconds]
training_set = train_set
testing_set = test_set

#
# Split data into n_timestamp
# 设置观察多久、预测多久
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
        seq_x = sequence[i:end_ix-pred_timestamp].reshape(n_timestamp,row*col)
        seq_y = sequence[end_ix-pred_timestamp:end_ix].reshape(pred_timestamp,row*col)
        # seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

X_train, y_train = data_split(training_set, n_timestamp,pred_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],row*col)
X_test, y_test = data_split(testing_set, n_timestamp, pred_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], row*col)
print("X_train shape:",X_train.shape)
print(X_train.shape[0], X_train.shape[1])


# 假设矩阵元素全0
y_predicted = np.zeros(shape=y_test.shape)

mse = mean_squared_error(y_test.reshape((y_test.shape[0],-1)), y_predicted.reshape(y_test.shape[0],-1))
# r2 = r2_score(y_test.reshape((y_test.shape[0],-1)), y_predicted.reshape(y_test.shape[0],-1))
mae= mean_absolute_error(y_test.reshape((y_test.shape[0],-1)), y_predicted.reshape(y_test.shape[0],-1))
print("mse=" + str(round(mse,2)))
# print("r2=" + str(round(r2,2)))
print("mae=" + str(round(mae,2)))


plt.figure(figsize=(8,7))
figId=1
for i in range(0,pred_timestamp,int(pred_timestamp/6+0.5)):  #+0.5实现四舍五入
    plt.subplot(3, 2, figId)
    plt.plot(y_test[0][i], color = 'black', linewidth=1, label = 'True')
    plt.plot(y_predicted[0][i], color = 'red', linewidth=1, label = 'Predicted')
    plt.legend(frameon=False)
    plt.ylabel("NUM of PERSON")
    plt.xlabel("GRID ID")
    plt.title("TIME %d"%i)
    figId=figId+1

plt.subplots_adjust(hspace = 0.5, wspace=0.3)
plt.show()