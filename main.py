from pontosNAS import PontosNAS
import math
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

def escolherAtividades():
    atividades = [random.choice(['1a', '1b', '1c']),
                  '2', '3',
                  random.choice(['4a', '4b', '4c']),
                  '5',
                  random.choice(['6a', '6b', '6c']),
                  random.choice(['7a', '7b']),
                  random.choice(['8a', '8b', '8c']),
                  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    return atividades


def executarAtendimento():
    resultados = {}
    for i in PontosNAS:
        try:
            aux = resultados[i]
        except KeyError:
            aux = []

        aux.append(stats.norm.rvs(PontosNAS[i], sigma))
        resultados[i] = aux

    print(resultados)


if __name__ == '__main__':

    # print(PontosNAS)
    media = PontosNAS['1a']
    variancia = 1

    sigma = math.sqrt(variancia)
    # = np.linspace(media - 3 * sigma, media + 3 * sigma, 100)
    #pdf = stats.norm.pdf(x_axis, media, sigma)
    #df = pd.DataFrame({'probability': pdf, 'x': x_axis})
    #sns.lineplot(data=df, x='x', y='probability')
    #plt.show()

    i=0
    while i<10:
    #    print(x_axis.item(i), "  =  ", pdf.item(i))
        #print(stats.norm.rvs(media, sigma))
    #    print(escolherAtividades())
        i=i+1

    executarAtendimento()

