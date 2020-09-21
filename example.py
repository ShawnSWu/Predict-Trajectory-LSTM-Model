from keras.models import load_model
import data_processor
import numpy as np

coordinate = np.array( [
    [22.902772, 120.273126, 15, 251],
    [22.902746, 120.273129, 15, 251],
    [22.902681, 120.273135, 15, 251],
    [22.902610, 120.273144, 15, 251]]
)

coordinate2 = np.array( [
    [22.902739, 120.272783, 19, 250],
    [22.902679, 120.272730, 19, 250],
    [22.902616, 120.272680, 19, 250],
    [22.902586, 120.272706, 19, 250]]
)

coordinate3 = np.array( [
    [22.9026133, 120.2730990, 16, 237],
    [22.9025968, 120.2730660, 16, 237],
    [22.9026111, 120.2730310, 16, 237],
    [22.9026310, 120.2729900, 16, 237]]
)

model = load_model( './uav_predict_model.h5', compile=False )

normalization_coordinate = data_processor.get_scaler().fit_transform( coordinate3 )

print(normalization_coordinate)

reshape_normalization_coordinate = np.reshape( normalization_coordinate, (1, normalization_coordinate.shape[0],
                                                                          normalization_coordinate.shape[1]) )

predict_normalization_coordinate = model.predict( reshape_normalization_coordinate )

predict_coordinate = data_processor.get_scaler().inverse_transform( predict_normalization_coordinate )

print(predict_coordinate)