# bit_inspect
import csv
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.layers import Conv2D, MaxPooling2D

yo_train_files = [r'C:\User\conyu\PycharmProject\bitinspect\yo_train.csv']
dou_train_files = [r'C:\User\conyu\PycharmProject\bitinspect\dou_train.csv']
yo_test_files = [r'C:\User\conyu\PycharmProject\bitinspect\yo_test.csv']
dou_test_files = [r'C:\User\conyu\PycharmProject\bitinspect\dou_test.csv']


batch_size = 128  # 每一轮训练中，一次输入的大小
num_classes = 2   # 二分类
epochs = 12       # 训练轮询次数
n = 60

input_shape = (n, 2)    # 输入维度


# 定义一个函数，用于从文件中读取数据，并返回数据和标记列表
def get_dataset(filei, cls):
    with open(filei) as f:
        reader = csv.DictReader(f)
        utilz = []
        for row in reader:
            utilz.append([float(row['Bytes']),float(row['All packets'])])
    x_data = [utilz[i:i+n] for i in range(0,(len(utilz)-n),1)]
    y_data = [cls]*len(x_data)
    return x_data, y_data


for filei in yo_train_files:
    yo_x_train, yo_y_train = get_dataset(filei, 0)

for filei in dou_train_files:
    dou_x_train, dou_y_train = get_dataset(filei, 1)

import random

x_train = yo_x_train + dou_x_train
y_train = yo_y_train + dou_y_train
index = [i for i in range(len(x_train))]
# 随机排列列表index
random.shuffle(index)

data = []
label = []
for i in index:
    data.append(x_train[i])
    label.append(y_train[i])
x_train = data
y_train = label

x_train = numpy.array(x_train)
print(x_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train.shape)

for filei in yo_test_files:
    yo_x_test, yo_y_test = get_dataset(filei, 0)

for filei in yo_test_files:
    dou_x_test, dou_y_test = get_dataset(filei, 1)

x_test = yo_x_test + dou_x_test
y_test = yo_y_test + dou_y_test

x_test = numpy.array(x_test)
print(x_test.shape)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)

from keras.layers import LSTM, Flatten

model = Sequential() # 得到神经网络模型

# 向神经网络模型加层
model.add(Conv1D(32, 4, input_shape=input_shape))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))  # 加入全连接层
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))  # 激活函数sigmoid,多分类用softmax

model.compile(loss=keras.losses.categorical_crossentropy,    # loss函数
              optimizer=keras.optimizers.Adadelta(),         # 优化器
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,    # 输入大小
          epochs=epochs,            # 训练轮询次数
          verbose=1,
          shuffle=True,
          validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test, verbose=0)  # 将模型输出数据与标记作比较，评估模型
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存模型
model.save(r'C:\User\conyu\Desktop\model_data\my_model.h5')
