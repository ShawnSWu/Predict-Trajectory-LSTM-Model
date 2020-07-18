# -*- coding: utf-8 -*-
import LSTMModel as utm_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# 每四個點預測下一個點
train_size, predict_size = 4, 1

# 要用來訓練的資料欄位，只取特定欄位，總共10個
data_column = ['lat', 'lon', 'x_gyro', 'y_gyro', 'z_gyro', 'x_acc', 'y_acc', 'z_acc', 'wind_speed', 'wind_direction']

df = pd.read_csv( 'data/06242020_133209.csv' )
dataset = df.loc[:, data_column].values

# 把資料都轉MinMax
sc = MinMaxScaler( feature_range=(0, 1) )
training_set_scaled = sc.fit_transform( dataset )


# 把csv檔的時間序資料切成每4筆預測下一筆，每四筆資料存在x, 而y就是label(每四筆的下一筆)
x, y = utm_model.train_windows( training_set_scaled, train_size, predict_size )

# 切train test data
train_x, train_y, test_x, test_y = train_test_split( x, y, test_size=0.1, random_state=0 )


# 要轉成3維才能丟入keras的LSTM中，但我train_x, train_y應該已經是三維了，但是這樣跑會出現
# ValueError: Error when checking target: expected activation_1 to have 2 dimensions, but got array with shape (2, 4, 9)的錯誤

print(train_x.shape)
print(train_y.shape)


model = utm_model.trainModel( train_x, train_y )
loss, acc = model.evaluate( test_x, test_y, verbose=2 )
print('Loss : {}, Accuracy: {}'.format( loss, acc * 100 ))