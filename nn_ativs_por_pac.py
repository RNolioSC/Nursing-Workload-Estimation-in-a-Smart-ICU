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
import keras.metrics
from sklearn.preprocessing import normalize
import csv
import tensorflow as tf


def data_preprocessing():
    with open("ativs_diag.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers

        tabela = [row for row in reader]

    diagnosticos = []
    nova_tabela = []
    for linha in tabela:
        nova_linha = linha
        if linha[2] not in diagnosticos:
            diagnosticos.append(linha[2])
        nova_linha[2] = diagnosticos.index(linha[2])
        nova_tabela.append(nova_linha)

    with open('ativs_diag_prepr.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in nova_tabela:
            filewriter.writerow(i)


if __name__ == '__main__':
    tempoi = time.time()

    data_preprocessing()
    dataset = genfromtxt(r'ativs_diag_prepr.csv', encoding='latin-1', delimiter=',', skip_header=2)
    dataset = dataset[..., 2:]  # drop codPaciente, Dia

    data_norm = normalize(dataset, axis=0, norm='max')
    X = data_norm[:, 0]
    Y = data_norm[:, 1:]
    # for i in range(0, len(X)):
    #     for j in range(0, len(X[0])):
    #         if numpy.isnan(X[i][j]):
    #             print("NaN: ", i, j)

    #####

    model = Sequential()
    model.add(Dense(150, input_dim=1, activation='tanh'))
    model.add(Dense(150, activation='tanh'))
    model.add(Dense(23, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X, Y, epochs=10, batch_size=10)

    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy * 100))

    # predictions = model.predict_classes(X)  # deprecated
    predict_y = model.predict(X)
    classes_y = numpy.argmax(predict_y, axis=1)


    tempof = time.time()
    print("tempo de execucao (s):", tempof - tempoi)

    # model.save('modelo.H5')

    # graficos de acuracia e validacao
    plt.plot(history.history['accuracy'])
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    # plt.legend(['Treinamento'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.ylabel('Perda')
    plt.xlabel('Época')
    # plt.legend(['Treinamento'])
    plt.show()

    # print("False positives: ", keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None).result().numpy())
    #matrix = confusion_matrix(Y, predict_y)
    #print(matrix)

    #plt.matshow(tf.math.confusion_matrix(Y[0], predict_y[0]))
    #plt.show()
    # fp = matrix[1, 0]
    #
    # print('False positives =', (fp/numpy.size(X))*100, '%')
    # print('x[0].s=', numpy.size(X[0]))

    # loss: 124.5392 - accuracy: 0.7756 - elu
    # loss: 16.7161 - accuracy: 0.7756 - tanh

    # TODO: comparacao pontos NAS totais calc pela nn vs teoricos pelos papers vs simulados.