from keras.models import load_model
import data_processor as dp
import numpy as np

coordinate = np.array(
 [
  [ 22.90260309, 120.27294747,  20.0, 260.0],
  [ 22.90260331, 120.27298366, 20.0, 260.0],
  [ 22.90260474, 120.27302164,  20.0, 260.0],
  [ 22.90260712, 120.27305682,  20.0, 260.0]
 ])

model = load_model('./uav_predict_model.h5', compile=False)

normalization_coordinate = dp.get_scaler().fit_transform(coordinate)

reshape_normalization_coordinate = np.reshape(normalization_coordinate, (1, normalization_coordinate.shape[0], normalization_coordinate.shape[1]))

predict_normalization_coordinate = model.predict(reshape_normalization_coordinate)

predict_coordinate = dp.get_scaler().inverse_transform(predict_normalization_coordinate)

print(predict_coordinate)