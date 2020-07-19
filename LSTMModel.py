from keras.layers.core import Activation
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

data_column = ['lat', 'lon']


def get_train_odel(train_X, train_Y):
    model = Sequential()
    model.add( LSTM(
        120,
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True ) )
    model.add( Dropout( 0.3 ) )

    model.add( LSTM(
        120,
        return_sequences=False ) )
    model.add( Dropout( 0.3 ) )

    model.add( Dense(
        train_Y.shape[1] ) )
    model.add( Activation( "relu" ) )

    model.compile( loss='mse', optimizer='adam', metrics=['acc'] )
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
