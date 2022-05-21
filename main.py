from pontosNAS import PontosNAS
import math
import scipy.stats as stats
import random
import datetime
from Atendimento import Atendimento
import csv


def escolher_atividades():
    atividades = [random.choice(['1a', '1b', '1c']),
                  '2', '3',
                  random.choice(['4a', '4b', '4c']),
                  '5',
                  random.choice(['6a', '6b', '6c']),
                  random.choice(['7a', '7b']),
                  random.choice(['8a', '8b', '8c']),
                  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    return atividades


def simular_nas(sigma, atividades):
    resultados = {}
    for j in atividades:
        aux = stats.norm.rvs(PontosNAS[j], sigma)
        resultados[j] = aux
    return resultados


def exportar_antendimentos(atendimentos):
    with open('atendimentos.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Paciente', 'Dia', 'Horario', 'Atividade', 'Pontuacao'])

        for atendimento in atendimentos:
            aux = [atendimento.get_paciente_str(), atendimento.get_dia_str(), atendimento.get_horario_str(),
                   atendimento.get_atividade_str(), atendimento.get_pontuacao_str()]

            filewriter.writerow(aux)

    return


if __name__ == '__main__':

    variancia = 1
    sigma = math.sqrt(variancia)

    pacientes = ['p1', 'p2', 'p3']
    data_inicio = datetime.datetime(year=2022, month=1, day=1)
    total_dias = 2

    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio+datetime.timedelta(i))

    atendimentos = []
    for paciente in pacientes:
        for dia in dias:
            atividades = escolher_atividades()
            lista_nas = simular_nas(sigma, atividades)

            for atividade in atividades:
                atendimento = Atendimento(paciente, dia, atividade, lista_nas[atividade])
                atendimentos.append(atendimento)

    #print(atendimentos)
    exportar_antendimentos(atendimentos)
