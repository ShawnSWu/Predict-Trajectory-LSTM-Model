import keras as keras
from keras.layers.core import Activation, Dropout
from keras.layers import LSTM, Dense
from keras.models import Sequential
import keras.backend as K


def get_train_odel(train_X, train_Y):
    model = Sequential()
    model.add( LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True ))
    model.add( Dropout( 0.3 ) )

    model.add( LSTM(128, return_sequences=False ) )
    model.add( Dropout( 0.3 ) )

    model.add( Dense(train_Y.shape[1]))
    model.add( Activation( "relu" ) )

    opt = keras.optimizers.Adam( learning_rate=0.001 )
    model.compile( loss=regularization_mse_loss_function, optimizer=opt, metrics=[f1_score])
    model.summary()
    return model



def window_data(data, window_size):
    X = []
    y = []
    i = 0
    while (i + window_size) <= len( data ) - 1:
        X.append( data[i:i + window_size] )
        y.append( data[i + window_size] )
        i += 1
    assert len( X ) == len( y )
    return X, y



def regularization_mse_loss_function(y_true, y_pre, alpha=0.1):
    mse = keras.losses.mean_squared_error( y_true, y_pre )
    return mse + ( L2_loss(y_true, y_pre ) * alpha)


def L1_loss(y_true,y_pre):
    return K.sum(K.abs(y_true-y_pre))


def L2_loss(y_true,y_pre):
    return K.sum(K.square(y_true-y_pre))


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))