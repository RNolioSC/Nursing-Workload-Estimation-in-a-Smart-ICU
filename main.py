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


def escolher_atividades_baixo_risco():
    atividades = ['1a', '2', '3', '4a', '5', '6a', '7a', '8a',
                  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    return atividades


def escolher_atividades_medio_risco():
    atividades = ['1b', '2', '3', '4b', '5', '6b', '7b', '8b',
                  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    return atividades


def escolher_atividades_alto_risco():
    atividades = ['1c', '2', '3', '4c', '5', '6c', '7b', '8c',
                  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    return atividades


def simular_nas(sigma, atividades):
    resultados = {}
    for j in atividades:
        aux = abs(stats.norm.rvs(PontosNAS[j], sigma))
        resultados[j] = aux
    return resultados


def exportar_antendimentos(atendimentos):
    with open('atendimentos.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Paciente', 'Dia Inicio', 'Horario Inicio', 'Dia Fim', 'Horario Fim',
                             'Atividade', 'Pontuacao', 'Enfermeiro', 'Tecnico'])

        for atendimento in atendimentos:
            aux = [atendimento.get_paciente_str(), atendimento.get_dia_inicio_str(),
                   atendimento.get_horario_inicio_str(),
                   atendimento.get_dia_fim_str(), atendimento.get_horario_fim_str(),
                   atendimento.get_atividade_str(),
                   atendimento.get_pontuacao_str()]

            if atendimento.get_enfermeiro() is None:
                aux.append('')
            else:
                aux.append(atendimento.get_enfermeiro().get_codigo())

            if atendimento.get_tecnico() is None:
                aux.append('')
            else:
                aux.append(atendimento.get_tecnico().get_codigo())

            filewriter.writerow(aux)

    return


def gerar_cod_rfid(tipo):

    rfid = ''
    if tipo == "enfermeiro":
        rfid = '0'
    elif tipo == "tecnico":
        rfid = '1'
    else:
        pass

    for j in range(7):
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
        rfid = gerar_cod_rfid("enfermeiro")
        nome = 'enfermeiro' + str(j+1)
        enfermeiros.append(Enfermeiro(rfid, nome, 'enfermeiro'))

    return enfermeiros


def simular_tecnicos(quantidade):
    tecnicos = []

    for j in range(quantidade):
        rfid = gerar_cod_rfid("tecnico")
        nome = 'tecnico' + str(j+1)
        tecnicos.append(Enfermeiro(rfid, nome, 'tecnico'))

    return tecnicos


def simular_atendimentos():
    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio_sim + datetime.timedelta(i))

    atendimentos = []
    for paciente in pacientes:

        for dia in dias:
            atividades = escolher_atividades()
            lista_nas = simular_nas(sigma, atividades)
            enfermeiro = random.choice(enfermeiros)
            tecnico = random.choice(tecnicos)

            data_inicio = dia + datetime.timedelta(minutes=random.randint(0, 30))
            data_fim = data_inicio

            for atividade in atividades:

                # atendimento comeca depos do fim do anterior
                data_inicio = data_fim + datetime.timedelta(minutes=random.randint(0, 10))
                data_fim = data_inicio + datetime.timedelta(minutes=pontos_to_minutos(lista_nas[atividade]))

                if atividade in ['7a', '7b', '8a', '8b']:  # somente o enfermeiro
                    atendimento = Atendimento(paciente, data_inicio, data_fim, atividade, lista_nas[atividade],
                                              enfermeiro, None)
                else:  # enfermeiro, tecnico ou ambos
                    aux = random.randint(0, 3)
                    if aux == 0:  # somente o enfermeiro
                        atendimento = Atendimento(paciente, data_inicio, data_fim, atividade, lista_nas[atividade],
                                                  enfermeiro, None)
                    elif aux == 1:  # somente o tecnico
                        atendimento = Atendimento(paciente, data_inicio, data_fim, atividade, lista_nas[atividade],
                                                  None, tecnico)
                    else:  # enfermeiro e tecnico
                        data_fim = data_inicio + datetime.timedelta(minutes=pontos_to_minutos(lista_nas[atividade])/2)
                        atendimento = Atendimento(paciente, data_inicio, data_fim, atividade, lista_nas[atividade],
                                                  enfermeiro, tecnico)


                atendimentos.append(atendimento)
    return atendimentos


def exportar_enfermeiros():
    with open('enfermeiros.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Codigo RFID', 'Nome', 'Tipo'])

        for enfermeiro in enfermeiros:
            aux = [enfermeiro.get_codigo(), enfermeiro.get_nome(), enfermeiro.get_tipo()]

            filewriter.writerow(aux)
        for tecnico in tecnicos:
            aux = [tecnico.get_codigo(), tecnico.get_nome(), tecnico.get_tipo()]

            filewriter.writerow(aux)
    return


def simular_pacientes(quantidade):
    pacientes = []
    for j in range(quantidade):
        nome = 'paciente' + str(j + 1)
        pacientes.append(nome)

    return pacientes


def pontos_to_minutos(pontos):
    return pontos * 14.4


def minutos_to_pontos(minutos):
    return minutos / 14.4


if __name__ == '__main__':

    variancia = 1
    sigma = math.sqrt(variancia)

    pacientes = simular_pacientes(3)
    data_inicio_sim = datetime.datetime(year=2022, month=1, day=1)
    total_dias = 5

    enfermeiros = simular_enfermeiros(7)
    tecnicos = simular_tecnicos(4)

    exportar_enfermeiros()
    atendimentos = simular_atendimentos()
    exportar_antendimentos(atendimentos)
    # TODO: adicionar turnos de trabalho dos enfermeiros e tecnicos
    # TODO: gerar csv com tag enfermeiro, nome e total de pontos NAS por dia