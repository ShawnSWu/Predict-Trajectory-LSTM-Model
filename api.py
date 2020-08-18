from flask import Flask, request, jsonify
from tensorflow_core.python.keras.models import load_model
import data_processor
import numpy as np

app = Flask(__name__)


model = load_model('./uav_predict_modelF.h5', compile=False)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_input = request.get_json()
        trajectory = model_input['model_input']
        np_trajectory = np.array(trajectory)
        normalization_coordinate = data_processor.get_scaler().fit_transform(np_trajectory)

        print(normalization_coordinate)
        reshape_normalization_coordinate = np.reshape(normalization_coordinate, (1, normalization_coordinate.shape[0],
                                                                                 normalization_coordinate.shape[1]))
        predict_normalization_coordinate = model.predict(reshape_normalization_coordinate)
        predict_coordinate = data_processor.get_scaler().inverse_transform(predict_normalization_coordinate)
        print(predict_coordinate)

        lat = predict_coordinate[0][0]
        lon = predict_coordinate[0][1]

        predict_coordinate = '{%s, %s}' % (str(lat), str(lon))
        print(predict_coordinate)
    return predict_coordinate



if __name__ == "__main__":
    app.run()