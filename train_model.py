# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import predict_trajectory_model as utm_model
import data_processor as data_processor
from sklearn.model_selection import train_test_split
import numpy as np


#取得所有csv檔
all_csv_file = data_processor.get_all_csv_file_list(r'data/DroneFlightData/WithoutTakeoff')

# 取得每一隻csv檔的train_windows資料的集合
train_data, label = data_processor.get_all_train_data_and_label_data(all_csv_file)

# 切分訓練集, 測試集
x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size=0.1, random_state=0)

print(x_train.shape)

model = utm_model.get_train_odel( x_train, y_train )
                                                                                # 驗證集 split
result = model.fit( x_train, y_train, epochs=1, batch_size=50, verbose=1, validation_split=0.1 )
loss, f1_score = model.evaluate( x_test, y_test, verbose=2 )

# coordinate_x_test = data_processor.get_scaler().inverse_transform(x_test[4])
# print(coordinate_x_test)
#
# a = data_processor.get_scaler().inverse_transform(x_test[5])
# print(a)
#
# b = data_processor.get_scaler().inverse_transform(x_test[6])
# print(b)
#
# print("=======================")
#
#
#
# coordinate_y_test = data_processor.get_scaler().inverse_transform(y_test)
# print(coordinate_y_test[4])
# print(coordinate_y_test[5])
# print(coordinate_y_test[6])
#
#
# print("=======================")
#
#
# predict_coordinate = model.predict(x_test)
# predict_coordinate_inverse = data_processor.get_scaler().inverse_transform(predict_coordinate)
#
# print(predict_coordinate_inverse[4])
# print(predict_coordinate_inverse[5])
# print(predict_coordinate_inverse[6])



# model.save( "./uav_predict_model.h5" )

print(loss, f1_score)

plt.plot( result.history['loss'] )
plt.plot( result.history['val_loss'] )
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
