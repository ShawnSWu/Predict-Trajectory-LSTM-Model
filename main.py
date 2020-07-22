# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import LSTMModel as utm_model
import DataProcessor as data_processor
from sklearn.model_selection import train_test_split


#取得所有csv檔
all_csv_file = data_processor.get_all_csv_file_list(r'data/DroneFlightData/WithoutTakeoff')

# 取得每一隻csv檔的train_windows資料的集合
train_data, label = data_processor.get_all_train_data_and_label_data(all_csv_file)


# 切分訓練集, 測試集
x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size=0.1, random_state=0)

print(x_train.shape)

model = utm_model.get_train_odel( x_train, y_train )
                                                                                # 驗證集 split
result = model.fit( x_train, y_train, epochs=1000, batch_size=50, verbose=1, validation_split=0.2 )
loss, f1_score = model.evaluate( x_test, y_test, verbose=2 )


plt.plot( result.history['loss'] )
plt.plot( result.history['val_loss'] )
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
