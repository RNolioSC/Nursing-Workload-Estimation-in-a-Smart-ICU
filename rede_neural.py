from numpy import genfromtxt
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
import time
import keras
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    
    tempoi = time.time()

    #### TODO: data preprocessing

    dataset = genfromtxt(r'atendimentos_nn.csv', encoding='latin-1', delimiter=',', skip_header=1)
    X = dataset[:, 1:]
    Y = dataset[:, 0]
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 1-1")

    #####

    model = Sequential()
    model.add(Dense(25, input_dim=3, activation='softsign'))
    model.add(Dense(25, activation='softsign'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    history = model.fit(X, Y, epochs=10, batch_size=100)

    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy * 100))

    predictions = model.predict_classes(X)
    
    tempof = time.time()
    print("tempo de execucao (s):", tempof-tempoi)
   
    #model.save('modelo.H5')


    #graficos de acuracia e validacao
    plt.plot(history.history['accuracy'])
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treinamento'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treinamento'])
    plt.show()
    
    print("False positives: ", keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None).result().numpy())
    matrix = confusion_matrix(Y, predictions)
    print(matrix)
    fp = matrix[1, 0]
    
    print('False positives =', (fp/numpy.size(X))*100, '%')
    #print('x[0].s=', numpy.size(X[0]))
    
    
    
    
