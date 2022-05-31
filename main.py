from pontosNAS import PontosNAS
import math
import scipy.stats as stats
import random
import datetime
from Atendimento import Atendimento
import csv
from Enfermeiro import Enfermeiro


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

        filewriter.writerow(['Paciente', 'Dia', 'Horario', 'Atividade', 'Pontuacao', 'Enfermeiro'])

        for atendimento in atendimentos:
            aux = [atendimento.get_paciente_str(), atendimento.get_dia_str(), atendimento.get_horario_str(),
                   atendimento.get_atividade_str(), atendimento.get_pontuacao_str(),
                   atendimento.get_enfermeiro().get_codigo()]

            filewriter.writerow(aux)

    return


def gerar_cod_rfid():
    rfid = ''
    for j in range(8):
        aux = random.randint(0, 15)
        if aux == 15:
            aux_str = 'F'
        elif aux == 14:
            aux_str = 'E'
        elif aux == 13:
            aux_str = 'D'
        elif aux == 12:
            aux_str = 'C'
        elif aux == 11:
            aux_str = 'B'
        elif aux == 10:
            aux_str = 'A'
        else:
            aux_str = str(aux)
        rfid = rfid + aux_str
    return rfid


def simular_enfermeiros(quantidade):
    enfermeiros = []

    for j in range(quantidade):
        rfid = gerar_cod_rfid()
        nome = 'enfermeiro' + str(j+1)
        enfermeiros.append(Enfermeiro(rfid, nome))

    return enfermeiros


def simular_atendimentos():
    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio + datetime.timedelta(i))

    atendimentos = []
    for paciente in pacientes:

        for dia in dias:
            atividades = escolher_atividades()
            lista_nas = simular_nas(sigma, atividades)
            enfermeiro = random.choice(enfermeiros)

            for atividade in atividades:
                atendimento = Atendimento(paciente, dia, atividade, lista_nas[atividade], enfermeiro)
                atendimentos.append(atendimento)

    # print(atendimentos)
    return atendimentos


def exportar_enfermeiros():
    with open('enfermeiros.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Codigo RFID', 'Nome'])

        for enfermeiro in enfermeiros:
            aux = [enfermeiro.get_codigo(), enfermeiro.get_nome()]

            filewriter.writerow(aux)
    return


if __name__ == '__main__':

    variancia = 1
    sigma = math.sqrt(variancia)

    pacientes = ['p1', 'p2', 'p3']
    data_inicio = datetime.datetime(year=2022, month=1, day=1)
    total_dias = 2

    enfermeiros = simular_enfermeiros(7)
    #for enfermeiro in enfermeiros:
    #    print(enfermeiro)

    exportar_enfermeiros()
    atendimentos = simular_atendimentos()
    exportar_antendimentos(atendimentos)
