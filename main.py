# 0 = non- rumor , 1 = rumor

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,LSTM,GRU,Bidirectional
# adding temporal cnn
from tcn import TCN, tcn_full_summary
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers


models_dir = 'D:/BOOKS/twitterRumor/models'

def load_data(event,time=60):
    events = ["charliehebdo", "ferguson", "germanwings-crash", "gurlitt", "ottawashooting", "putinmissing",
                "sydneysiege"]
    data = pd.read_csv(f'CSV_Files/{events[event]}.csv', names=['timeDiff', 'status', 'Freq'], header=0)
    
    return data

def split_train_test(training):
    x_train = training[['timeDiff','Freq']]
    scaler =  MinMaxScaler(feature_range=(0,1))
    x_train = scaler.fit_transform(training[['status']].values.reshape(-1, 1))
    train_size = int(len(x_train) * 0.7)
    test_size = len(x_train) - train_size
    train, test = x_train[0:train_size,:], x_train[train_size:len(x_train),:]

    return train,test



def create_dataset(x_train, look_back=1):
    dataX, dataY = [], []
    for i in range(len(x_train)-look_back-1):
        a = x_train[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(x_train[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



def build_tcnn(x_train,y_train,save=False):
    look_back=1
    model = Sequential() 
    model.add(TCN(input_shape=(x_train.shape[1], look_back)) )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',metrics=["accuracy"])

    tcn_full_summary(model, expand_residual_blocks=False)
    model.fit(x_train, y_train, epochs=5)#, validation_split=0.2

    if save == True: 
        model.save(f'{models_dir}/TCNN')
    return model




def build_LSTM(x_train,y_train,save=False):
    #,x_test, y_test
    look_back = 1
    model = Sequential()
    model.add(LSTM(256, return_sequences = True, input_shape = (x_train.shape[1], look_back)))
    model.add(LSTM(128,input_shape = (x_train.shape[1], look_back)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile( loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=2)
    
    if save == True: 
        model.save(f'{models_dir}/LSTM')
    return model



def build_RNN(x_train,y_train,save=False):
    
    model = Sequential()
    model.add(layers.Embedding(input_dim=100, output_dim=64))
    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(SimpleRNN(128))
    model.add(Dense(1,activation='sigmoid'))
    #sigmoid added- 1 unit
    model.summary()

    model.compile( loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    
    model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=2)

    if save==True:        
        model.save(f'{models_dir}/SimpleRNN')
    return model






def build_GRU(x_train, y_train,save=False):

    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=100, output_dim=64))
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(SimpleRNN(128))

    model.add(Dense(1,activation='sigmoid'))

    model.summary()
    
    model.compile( loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=2)
    
    if save==True:
        model.save(f'{models_dir}/GRU')
    return model


def build_bi_LSTM(x_train, y_train,save=False):
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(5, 1)) )
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    model.compile( loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=2)
    # epochs=20 changing to 30

    if save==True:
        model.save(f'{models_dir}/BiLSTM')
    return model



def load_models():
    # loading mdoel
    models = []
    models.append(keras.models.load_model(f'{models_dir}/LSTM'))
    models.append(keras.models.load_model(f'{models_dir}/SimpleRNN') )
    models.append(keras.models.load_model(f'{models_dir}/GRU'))
    models.append(keras.models.load_model(f'{models_dir}/BiLSTM'))
    models.append(keras.models.load_model(f'{models_dir}/TCNN'))
    return models
    


def Ensembler_result(models,x_test):
    array = None
    for model in models:
        if array is None:
            array = model.predict(x_test)
        else:
            a = model.predict(x_test)
            array = np.append(array,a, axis=1)
    
    result = []
    for row in iter(array):
        if (row >= 0.50).sum() >=3:
            result.append(1)
        else:
            result.append(0)
        
    return np.array(result)



if __name__ == "__main__":

    #["charliehebdo", "ferguson", "germanwings-crash", "gurlitt", "ottawashooting", "putinmissing","sydneysiege"]
    data = load_data(0)

    train, test = split_train_test(data)
    
    #print('train shape = ',train.shape)
    #print('test shape = ',test.shape)
    
    x_train, y_train = create_dataset(train, look_back=1)
    x_test, y_test = create_dataset(test, look_back=1)
    
    # reshaping
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    # shapes
    
    #print('x train shape=',x_train.shape)
    #print('y train shape=',y_train.shape)
    #print('x test shape=',x_test.shape)
    #print('y test shape=',y_test.shape)
    #print(x_train)
    #print(x_test)
    
    # Running and Saving Models
    model1 = build_LSTM(x_train,y_train,False)
    model2 = build_GRU(x_train,y_train,False)
    model3 = build_bi_LSTM(x_train,y_train,False)
    model4 = build_RNN(x_train,y_train,False)    
    model5 = build_tcnn(x_train,y_train,False)


    load_models = False
    
    if load_models == True:
        models = load_models()
    
    else: # use above build models 
        models = []
        models.append(model1)
        models.append(model2)
        models.append(model3)
        models.append(model4)
        models.append(model5)


    result =  Ensembler_result(models,x_test)

    r = (result == y_test).sum()
    acc = r/len(y_test)
    print('Non Rumor',(result == 0).sum() )

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_test,result)
    #labels=['NonRumor','Rumor']
    print('===='*10)
    print(matrix)
    print('Non-Rumor')
    print('===='*10)
    
    TP = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TN = matrix[1][1]

    accuracy = (TN+TP) /(TN+TP+FN+FP)
    precision = TP / (TP+FP)
    recall = TP /(TP+FN)

    f1_score = 2*((precision * recall)/(precision + recall))

    print('accuracy = ',accuracy)
    print('precision = ',precision)
    print('recall',recall)
    print('f1 score',f1_score)

    print('===='*10)
    print('HURRAAAHHH !!!!\n accuracy = ',acc)
    print('===='*10)
    
    
    