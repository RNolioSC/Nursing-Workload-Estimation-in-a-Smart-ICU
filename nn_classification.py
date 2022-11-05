from numpy import genfromtxt
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import time
import keras
import keras.metrics
from sklearn.preprocessing import normalize
import csv
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


def save_history(history):
    with open('modelo_classificacao/history.bin', 'wb') as f:
        pickle.dump(history.history, f)


def load_history():
    with open('modelo_classificacao/history.bin', 'rb') as f:
        history = pickle.load(f)
    return history


if __name__ == '__main__':

    treinar = False
    # Treinamento da rede neural
    tempoi = time.time()

    # path pros dados a serem usados para o treinamento
    simulacao_path = 'simulacoes/simulacao1'
    if treinar:
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

        model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['Recall'], loss_weights=list(PontosNAS.values()))
        history = model.fit(X, Y, epochs=50, batch_size=100, validation_split=0.2)
        save_history(history)
        history = history.history

        _, accuracy = model.evaluate(X, Y)
        print('Accuracy: %.2f' % (accuracy * 100))
        model.save('modelo_classificacao')

    else:  # nao treinar
        model = keras.models.load_model('modelo_classificacao')
        history = load_history()
        _, accuracy = model.evaluate(X, Y)
        print('Recall: %.2f' % (accuracy * 100))

    predict_y = model.predict(X)
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

    # graficos de acuracia e validacao
    # noinspection PyUnboundLocalVariable
    plt.plot(history['recall'])
    plt.plot(history['val_recall'])
    plt.ylabel('Recall')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Teste'])
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Teste'])
    plt.show()

    for i in range(0, len(predict_y)):
        for j in range(0, len(predict_y[0])):
            if predict_y[i][j] > 0.5:
                predict_y[i][j] = 1
            else:
                predict_y[i][j] = 0
