import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential


import numpy as np

data_column = ['lat', 'lon']


def trainModel(train_X, train_Y):
    model = Sequential()
    model.add( LSTM( 32, input_shape=(train_X.shape[1], train_X.shape[2]) ) )
    model.add( Dropout( 0.3 ) )

    model.add( Dense( train_Y.shape[1] ) )
    model.add( Activation( "relu" ) )

    model.compile( loss='mse', optimizer='adam', metrics=['acc'] )
    model.fit( train_X, train_Y, epochs=100, batch_size=64, verbose=1)
    model.summary()

    return model


def train_windows(df, ref_point=4, predict_point=1):
    X_train, Y_train = [], []
    for i in range( df.shape[0] - predict_point - ref_point ):
        X_train.append( np.array( df[i:i + ref_point, :-1] ) )
        Y_train.append( np.array( df[i + ref_point:i + ref_point + predict_point] ) )
    return np.array( X_train ), np.array( Y_train )
