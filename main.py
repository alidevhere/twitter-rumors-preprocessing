import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,LSTM
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

    events = ["charliehebdo", "ferguson", "germanwings-crash", "gurlitt", "ottawashooting", "putinmissing",
              "sydneysiege"]

    li = []
    for e in events:
        li.append(f'D:/BOOKS/5th Sem/AI Lab/AI-PROJECT/Master_Thesis-master/processed/{e}.csv')
    l=[pd.read_csv(f,names=['timeDiff','status','Freq'],header=0) for f in li]
    data = pd.concat(l)
    #print(data)

    #print(data.shape)
    print(data.columns)
    x_data = data[['timeDiff','Freq']]
    y_data = data['status']
    y_data = to_categorical(y_data)


    # train the normalization
    scaler = MinMaxScaler()
    scaler = scaler.fit(x_data)
    # normalize the dataset and print
    x_data = scaler.transform(x_data)

    # split into  train and test
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=13)

    # reshaping to 3D
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # define model
    model = Sequential()
    model.add(LSTM(units=80, activation='relu', input_shape=(2,1) ))
    model.add(Dense(units=40,activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(optimizer='adam',loss = 'binary_crossentropy')
    print(model.summary())


    # fit model
    model.fit(x_train, y_train, epochs=60, verbose=2,batch_size=5, validation_split=0.2)
    # test_predict = model.predict(x_test)
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test,verbose=1)
    print("test loss, test acc:", results)


    '''
       with open(os.path.join('D:/BOOKS/5th Sem/AI Lab/AI-PROJECT/CSV_Files/DataDescription.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    '''

