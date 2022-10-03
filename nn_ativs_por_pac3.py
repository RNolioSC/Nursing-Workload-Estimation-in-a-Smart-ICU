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
import pickle
from pontosNAS import PontosNAS


def data_preprocessing():
    with open('bin/diagnosticos_list.bin', 'rb') as f:
        diagnosticos = pickle.load(f)
    with open("CSV/ativs_diag.csv") as csvfile:
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

    with open('CSV/ativs_diag_prepr2.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in nova_tabela:
            filewriter.writerow(i)


def diagnostico_str_to_float(diag):
    with open('bin/diagnosticos_list.bin', 'rb') as f:
        diagnosticos = pickle.load(f)
    return diagnosticos.index(diag) / len(diagnosticos)


def atividades_fl_to_str(atividades_fl):
    atividades = []
    if atividades_fl[0] < 1 / 3:
        atividades.append('1a')
    elif atividades_fl[0] < 2 / 3:
        atividades.append('1b')
    else:
        atividades.append('1c')
    if atividades_fl[1] > 0.5:
        atividades.append('2')
    if atividades_fl[2] > 0.5:
        atividades.append('3')

    if atividades_fl[3] < 1 / 3:
        atividades.append('4a')
    elif atividades_fl[3] < 2 / 3:
        atividades.append('4b')
    else:
        atividades.append('4c')
    if atividades_fl[4] > 0.5:
        atividades.append('5')

    if atividades_fl[5] < 1 / 3:
        atividades.append('6a')
    elif atividades_fl[5] < 2 / 3:
        atividades.append('6b')
    else:
        atividades.append('6c')

    if atividades_fl[6] < 0.5:
        atividades.append('7a')
    else:
        atividades.append('7b')

    if atividades_fl[7] < 1 / 3:
        atividades.append('8a')
    elif atividades_fl[7] < 2 / 3:
        atividades.append('8b')
    else:
        atividades.append('8c')

    for i in range(9, 23):
        if atividades_fl[i] > 0.5:
            atividades.append(str(i))
    return atividades


def evaluate_batch(diagnosticos):
    model = keras.models.load_model('modelo_nn')
    diagnosticos_fl = [diagnostico_str_to_float(d) for d in diagnosticos]
    atividades_fl = model.predict(numpy.array(diagnosticos_fl), verbose=0)
    #atividades_fl = model(numpy.array(diagnosticos_fl))
    #atividades_fl = numpy.array(atividades_fl).tolist()

    all_atividades = []
    for i in atividades_fl:
        all_atividades.append(atividades_fl_to_str(i))
    return all_atividades


def evaluate(diagnostico_str):
    model = keras.models.load_model('modelo_nn')
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

#
# def normalization(arr):
#     for i in arr.shape[0]:
#         for j in arr.shape[1]:
#             arr[i][j] = new_col.append(i/max(col))
#         new_arr.append(new_col)
#     return new_arr


if __name__ == '__main__':

    # Treinamento da rede neural

    tempoi = time.time()

    data_preprocessing()
    dataset = genfromtxt(r'CSV/ativs_diag_prepr2.csv', encoding='latin-1', delimiter=',')
    # dataset = dataset[..., 2:]  # drop codPaciente, Dia

    data_norm = normalize(dataset, axis=0, norm='max')
    #data_norm = dataset
    #tst = normalization(dataset)
    #
    # for i in range(0, 10):
    #     print(dataset[i, :])
    #     print(data_norm[i, :])
    #     print('')
    X = data_norm[:, 0]
    Y = data_norm[:, 1:]  # [i, j)
    #
    # for i in X:
    #     if i>1 or i<0:
    #         raise Exception
    # for j in Y:
    #     for k in j:
    #         if k>1 or k<0:
    #             raise Exception

    # quantidade de camadas e neuronios, pesquisar se tem como o otimizador fazer isso.
    model = Sequential()
    model.add(Dense(150, input_dim=1, activation='tanh'))  # testar com diferentes eh so um plus. relu funciona bem
    model.add(Dense(150, activation='tanh'))
    model.add(Dense(32, activation='sigmoid'))  # falar disto na discertacao, como desafio encontrado.
    # p/ tese: tanh as vezes nao converge (accuracy: 0.0426)
    #   relu, softplus, exponential: causa nan
    # selu = retorna alguns 1's. acc=0.24
    # relu eh boa quando tem muitos outliers
    # quantidade de dados pra simulacao: 10k dataset
    # fazer testes com varias funcoes de ativacao com o mesmo dataset
    # classes balanceadas: raro no mundo real.
    # uma ideia eh usar uma serie temporal, por turno. tem redes neurais projetadas para isto
    # mais importante: divisao treino / teste, crossvalidation com diferente distribuicao, aumentar tamanho simulacao,
    # # parte experimental, testar com novas  distribuicoes, balanceamento (dificil), series temporais
    # cross-validation nao muda mto se nao usar distribuicao diferente.
    # tcc de series temporarias: https://repositorio.ufsc.br/handle/123456789/223843
    # ciencia de dados: fica bom fazer analise descritiva de dados: media, mediana, quartis, formato dos dados:
    # por diagnostico. eh um diagrama de duracao por diagnostico: X=atividades nas, Y= duracao.
    # deixar claro que a escolha de atividades que a nn faz eh somente para a estimativa de num de enfermeiros, e nao
    # para decidir atividades a serem executadas para pacientes especificos

    #model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'], loss_weights=list(PontosNAS.values()))

    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['TruePositives'])
    # model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['BinaryAccuracy'])
    # https://stackoverflow.com/questions/65361359/why-the-accuracy-and-binary-accuracy-in-keras-have-same-result
    # causa nan: sgd
    # 14%: RMSprop, Adam, Nadam, Adamax. 38%: Adagrad, Ftrl
    # tf.keras.optimizers.Adam(learning_rate=0.1)
    # adadelta, adagrad
    history = model.fit(X, Y, epochs=30, batch_size=100, validation_split=0.2)

    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy * 100))

    # predictions = model.predict_classes(X)  # deprecated
    predict_y = model.predict(X)
    classes_y = []
    for i in predict_y:
        pass
    #classes_y = numpy.argmax(predict_y, axis=1)
    print(Y)
    print(predict_y)
    #print(classes_y)
    aux = model.predict([0, 1/3, 2/3, 3/3])
    aux2 = []
    for linha in aux:
        nova_linha = []
        for item in linha:
            if item < 0.5:
                nova_linha.append(0)
            else:
                nova_linha.append(1)
        aux2.append(nova_linha)


    # for i in range(0, len(aux)):
    #     for j in range(0, len(aux[i])):
    #         if aux[i][j] < 0.1:
    #             aux[i][j]=0
    print('predict diagnosticos:')
    print(aux2)
    #print('1...3', model.predict([1, 2, 3]))
    #model.layers.BatchNormalization(momentum=0.01)
    #predict_y = model(X, training=False)
    #
    # oldi = Y[0]
    # count = 0
    # for i in range(0, len(Y)):
    #     if Y[i][0] != oldi[0]:
    #         print('yy', Y[i])
    #         print('predicted', numpy.array(predict_y[i]).tolist())
    #         oldi = Y[i]
    #         count = count + 1
    #     if count > 20:
    #         break
    # plt.scatter(Y, [a for a in Y], '.')
    # plt.scatter(predict_y, [a for a in Y], 'o')
    # plt.legend('Yy', 'predicted')
    # plt.show()

    tempof = time.time()
    print("tempo de execucao (s):", tempof - tempoi)

    model.save('modelo_nn')
    # TODO: fazer treinamento com uma simulacao e validacao com outra simulacao, eh bem importante. cross dataset
    #  validation
    # conlcusoes: limitacao: independente da distribuicao, a rede vai aprender com aquela distribuicao ( normal). se
    # tiver uma simulacao com uma distribuicao pra treinamento, mas outra distribuicao pra testes. trabalhos futuros.
    # outra questao: durante a pandemia, eh outro cenario. oq mudou? se treinar com diferentes cenarios, fica melhor

    # graficos de acuracia e validacao
    # history.history: loss, binary_accuracy, val_los, val_binary_accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.ylabel('Acurácia')
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

    # ?: fix duracao com enfermeiro & tecnico
    # TODO: comparacao pontos NAS totais calc pela nn vs teoricos pelos papers vs simulados.