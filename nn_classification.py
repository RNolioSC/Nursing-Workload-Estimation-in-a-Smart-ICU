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
from sklearn.metrics import ConfusionMatrixDisplay
import keras.metrics
from sklearn.preprocessing import normalize
import csv
import tensorflow as tf
import pickle
from pontosNAS import PontosNAS


def data_preprocessing():
    with open(simulacao_path + '/bin/diagnosticos_list.bin', 'rb') as f:
        diagnosticos = pickle.load(f)
    with open(simulacao_path + '/CSV/ativs_diag.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        tabela = [row for row in reader]

    nova_tabela = []
    for linha in tabela:
        linha.pop(0)  # cod paciente
        linha.pop(0)  # dia
        nova_linha = [diagnosticos.index(linha.pop(0))]
        ativ_1 = linha.pop(0)
        if ativ_1 == '1':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_1 == '2':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_1 == '3':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        nova_linha.append(linha.pop(0))  # ativ 2 e 3
        nova_linha.append(linha.pop(0))

        ativ_4 = linha.pop(0)
        if ativ_4 == '1':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_4 == '2':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_4 == '3':
            nova_linha.append(1)
        else:
            nova_linha.append(0)

        nova_linha.append(linha.pop(0))  # ativ 5

        ativ_6 = linha.pop(0)
        if ativ_6 == '1':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_6 == '2':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_6 == '3':
            nova_linha.append(1)
        else:
            nova_linha.append(0)

        ativ_7 = linha.pop(0)
        if ativ_7 == '1':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_7 == '2':
            nova_linha.append(1)
        else:
            nova_linha.append(0)

        ativ_8 = linha.pop(0)
        if ativ_8 == '1':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_8 == '2':
            nova_linha.append(1)
        else:
            nova_linha.append(0)
        if ativ_8 == '3':
            nova_linha.append(1)
        else:
            nova_linha.append(0)

        nova_linha.extend(linha)
        nova_tabela.append(nova_linha)

    with open(simulacao_path + '/CSV/ativs_diag_prepr.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in nova_tabela:
            filewriter.writerow(i)


def diagnostico_str_to_float(diag, simulacao_path):
    with open(simulacao_path + '/bin/diagnosticos_list.bin', 'rb') as f:
        diagnosticos = pickle.load(f)
    return diagnosticos.index(diag) / (len(diagnosticos)-1)


def atividades_fl_to_str(atividades_fl):
    atividades_fl = list(atividades_fl)
    atividades_fl = [int(i) for i in atividades_fl]
    atividades = []
    if atividades_fl.pop(0) == 1:
        atividades.append('1a')
        atividades_fl.pop(0)
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('1b')
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('1c')
    else:
        raise Exception

    if atividades_fl.pop(0) == 1:
        atividades.append('2')
    if atividades_fl.pop(0) == 1:
        atividades.append('3')

    if atividades_fl.pop(0) == 1:
        atividades.append('4a')
        atividades_fl.pop(0)
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('4b')
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('4c')
    else:
        raise Exception

    if atividades_fl.pop(0) == 1:
        atividades.append('5')

    if atividades_fl.pop(0) == 1:
        atividades.append('6a')
        atividades_fl.pop(0)
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('6b')
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('6c')
    else:
        raise Exception

    if atividades_fl.pop(0) == 1:
        atividades.append('7a')
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('7b')
    else:
        raise Exception

    if atividades_fl.pop(0) == 1:
        atividades.append('8a')
        atividades_fl.pop(0)
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('8b')
        atividades_fl.pop(0)
    elif atividades_fl.pop(0) == 1:
        atividades.append('8c')
    else:
        raise Exception

    next_ativ = 9  # até 23
    while atividades_fl:
        if atividades_fl.pop(0) == 1:
            atividades.append(str(next_ativ))
        next_ativ += 1

    return atividades


def evaluate_batch(diagnosticos, simulacao_path):
    model = keras.models.load_model('modelo_classificacao')
    diagnosticos_fl = [diagnostico_str_to_float(d, simulacao_path) for d in diagnosticos]
    atividades_fl = model.predict(numpy.array(diagnosticos_fl), verbose=0)
    # atividades_fl = model(numpy.array(diagnosticos_fl))
    # atividades_fl = numpy.array(atividades_fl).tolist()

    for x in range(0, len(atividades_fl)):
        for y in range(0, len(atividades_fl[0])):
            if atividades_fl[x][y] < 0.5:
                atividades_fl[x][y] = 0
            else:
                atividades_fl[x][y] = 1

    all_atividades = []
    for i in atividades_fl:
        all_atividades.append(atividades_fl_to_str(i))
    return all_atividades


def evaluate(diagnostico_str):
    model = keras.models.load_model('modelo_classificacao')
    # um dado por vez:
    diagnostico_fl = diagnostico_str_to_float(diagnostico_str)
    atividades_fl = model(numpy.array([diagnostico_fl]))
    atividades_fl = numpy.array(atividades_fl).tolist()[0]

    # multiplos dados por vez: melhor performance

    # atividades_fl = model.predict([diagnostico_fl], batch_size=1, verbose=0).tolist()[0]
    # atividades_fl = model([diagnostico_fl], training=False).tolist()[0]
    # atividades_fl = model.predict_on_batch([diagnostico_fl]).tolist()[0]
    # atividades_fl = model([diagnostico_fl]).tolist()[0]
    # atividades_fl = model(tf.expand_dims([diagnostico_fl], axis=1).shape)
    #atividades_fl = model([[diagnostico_fl]])

    #atividades = atividades_fl_to_str(atividades_fl)

    return atividades_fl_to_str(atividades_fl)


if __name__ == '__main__':

    treinar = False
    # Treinamento da rede neural
    tempoi = time.time()

    # path pros dados a serem usados para o treinamento
    simulacao_path = 'simulacoes/simulacao1'

    data_preprocessing()
    dataset = genfromtxt(r'simulacoes/simulacao1/CSV/ativs_diag_prepr.csv', encoding='latin-1', delimiter=',')

    data_norm = normalize(dataset, axis=0, norm='max')

    X = data_norm[:, 0]
    Y = data_norm[:, 1:]  # [i, j)

    if treinar:
        model = Sequential()
        model.add(Dense(150, input_dim=1, activation='tanh'))  # testar com diferentes eh so um plus. relu funciona bem
        model.add(Dense(150, activation='tanh'))
        model.add(Dense(32, activation='sigmoid'))  # falar disto na discertacao, como desafio encontrado.

        # relu eh boa quando tem muitos outliers
        # fazer testes com varias funcoes de ativacao com o mesmo dataset
        # classes balanceadas: raro no mundo real.
        # mais importante: divisao treino / teste, crossvalidation com diferente distribuicao
        # # parte experimental, testar com novas  distribuicoes
        # cross-validation nao muda mto se nao usar distribuicao diferente.
        # ciencia de dados: fica bom fazer analise descritiva de dados: media, mediana, quartis, formato dos dados:
        # por diagnostico. eh um diagrama de duracao por diagnostico: X=atividades nas, Y= duracao.
        # TODO: adicionar diagrama mostrando a distribuicao de dados

        model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['Precision'], loss_weights=list(PontosNAS.values()))
        # precision: simply divides true_positives by the sum of true_positives and false_positives

        history = model.fit(X, Y, epochs=50, batch_size=100, validation_split=0.2)

        _, accuracy = model.evaluate(X, Y)
        print('Accuracy: %.2f' % (accuracy * 100))
        model.save('modelo_classificacao')

    else:  # nao treinar
        model = keras.models.load_model('modelo_classificacao')

    # predictions = model.predict_classes(X)  # deprecated
    predict_y = model.predict(X)
    # classes_y = numpy.argmax(predict_y, axis=1)
    print(Y)

    exemplo_x = [0, 1/3, 2/3, 3/3]
    aux = model.predict(exemplo_x)
    exemplo_y = []
    for linha in aux:
        nova_linha = []
        for item in linha:
            if item < 0.5:
                nova_linha.append(0)
            else:
                nova_linha.append(1)
        exemplo_y.append(nova_linha)

    print('predict diagnosticos:')
    print(exemplo_y)

    tempof = time.time()
    print("tempo de execucao (s):", tempof - tempoi)

    # TODO: fazer treinamento com uma simulacao e validacao com outra simulacao, eh bem importante. cross dataset
    #  validation
    # conlcusoes: limitacao: independente da distribuicao, a rede vai aprender com aquela distribuicao ( normal). se
    # tiver uma simulacao com uma distribuicao pra treinamento, mas outra distribuicao pra testes. trabalhos futuros.

    if treinar:  # TODO: save history
        # graficos de acuracia e validacao
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.ylabel('Verdadeiro Positivos')
        plt.xlabel('Época')
        plt.legend(['Treinamento', 'Teste'])
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('Perda')
        plt.xlabel('Época')
        plt.legend(['Treinamento', 'Teste'])
        plt.show()

    # print("False positives: ", keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None).result().numpy())
    for i in range(0, len(predict_y)):
        for j in range(0, len(predict_y[0])):
            if predict_y[i][j] > 0.5:
                predict_y[i][j] = 1
            else:
                predict_y[i][j] = 0
    # TODO: mostrar tabela com quantidade de falsos positivos/negativos.
    # matrix = confusion_matrix(Y.argmax(axis=1), predict_y.argmax(axis=1))
    # print(matrix)
    #
    # plt.matshow(tf.math.confusion_matrix(Y, predict_y))
    # plt.show()

    # confusion_matrix = confusion_matrix(Y.argmax(axis=1), predict_y.argmax(axis=1))
    #
    # cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    # cm_display.plot()
    # plt.show()
    # fp = matrix[1, 0]
    #
    # print('False positives =', (fp/numpy.size(X))*100, '%')
    # print('x[0].s=', numpy.size(X[0]))

