import Diagnosticos
from pontosNAS import PontosNAS
from ProbabsNas import ProbabsNAS
import math
import scipy.stats as stats
import random
import datetime
from Atendimento import Atendimento
import csv
from Enfermeiro import Enfermeiro
from Paciente import Paciente


def escolher_atividades(diagnostico):
    # ajustar probabilidade de cada atividade ocorrer de acordo com o diagnostico
    probNasDiag = Diagnosticos.Index[diagnostico]
    probsNASajustados = {}

    for k in ProbabsNAS:
        try:
            probsNASajustados[k] = probNasDiag[k]
        except KeyError:
            # nada a ajustar
            probsNASajustados[k] = ProbabsNAS[k]

    atividades = random.choices(['1a', '1b', '1c'],
                                weights=(probsNASajustados['1a'], probsNASajustados['1b'], probsNASajustados['1c']))

    for i in ['2', '3']:
        if probsNASajustados[i] >= random.random():
            atividades.append(i)

    atividades.extend(random.choices(['4a', '4b', '4c'],
                                     weights=(probsNASajustados['4a'], probsNASajustados['4b'], probsNASajustados['4c'])))

    for i in ['5']:
        if probsNASajustados[i] >= random.random():
            atividades.append(i)

    atividades.extend(random.choices(['6a', '6b', '6c'],
                                     weights=(probsNASajustados['6a'], probsNASajustados['6b'], probsNASajustados['6c'])))
    atividades.extend(random.choices(['7a', '7b'],
                                     weights=(probsNASajustados['7a'], probsNASajustados['7b'])))
    atividades.extend(random.choices(['8a', '8b', '8c'],
                                     weights=(probsNASajustados['8a'], probsNASajustados['8b'], probsNASajustados['8c'])))

    for i in ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']:
        if probsNASajustados[i] >= random.random():
            atividades.append(i)

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

        filewriter.writerow(['Paciente', 'Diagnostico', 'Dia Inicio', 'Horario Inicio', 'Dia Fim', 'Horario Fim',
                             'Atividade', 'Pontuacao', 'Enfermeiro', 'Tecnico'])

        for atendimento in atendimentos:
            aux = [atendimento.get_paciente_str(), atendimento.get_paciente().get_diagnostico(),
                   atendimento.get_dia_inicio_str(), atendimento.get_horario_inicio_str(),
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


def exportar_antendimentos_nn(atendimentos):
    with open('atendimentos_nn.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Diagnostico', 'codPaciente', 'Duracao',  #'DiaHoraInicio', 'DiaHoraFim',
                             'Atividade', 'Pontuacao', 'Enfermeiro', 'Tecnico'])

        for atendimento in atendimentos:
            duracao = atendimento.get_diaHoraFim() - atendimento.get_diaHoraInicio()

            aux = [atendimento.get_paciente().get_diagnostico(), atendimento.get_paciente().get_codigo(),
                   duracao.total_seconds(),
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


def simular_atendimentos():  # TODO: adicionar dias de folga
    variancia = 1
    sigma = math.sqrt(variancia)

    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio_sim + datetime.timedelta(i))

    atendimentos = []
    for paciente in pacientes:
        print('Paciente: ' + str(paciente))

        for dia in dias:
            atividades = escolher_atividades(paciente.get_diagnostico())
            lista_nas = simular_nas(sigma, atividades)

            enfermeiros_disponiveis = []
            for enfermeiro_aux in enfermeiros:
                if dia not in enfermeiro_aux.get_dias_trabalhados():
                    enfermeiros_disponiveis.append(enfermeiro_aux)
            if len(enfermeiros_disponiveis) == 0:
                raise Exception('Enfermeiros insuficientes!')

            tecnicos_disponiveis = []
            for tecnico_aux in tecnicos:
                if dia not in tecnico_aux.get_dias_trabalhados():
                    tecnicos_disponiveis.append(tecnico_aux)
            if len(tecnicos_disponiveis) == 0:
                raise Exception('Tecnicos insuficientes!')

            enfermeiro = random.choice(enfermeiros_disponiveis)
            enfermeiros_disponiveis.remove(enfermeiro)
            tecnico = random.choice(tecnicos_disponiveis)
            tecnicos_disponiveis.remove(tecnico)

            data_inicio_atividade = dia + datetime.timedelta(minutes=random.randint(0, 10))
            data_fim_atividade = data_inicio_atividade
            data_inicio_turno = dia

            prox_almoco = data_inicio_turno + datetime.timedelta(hours=(horas_turno/2))
            fim_turno = data_inicio_turno + datetime.timedelta(hours=horas_turno)

            for atividade in atividades:

                # atendimento comeca depois do fim do atendimento anterior
                data_inicio_atividade = data_fim_atividade + datetime.timedelta(minutes=random.randint(0, 10))

                if data_inicio_atividade > prox_almoco:  # pausa para o almoco
                    data_inicio_atividade = data_inicio_atividade + datetime.timedelta(minutes=30)
                    prox_almoco = prox_almoco + datetime.timedelta(hours=horas_turno)
                if data_inicio_atividade > fim_turno:  # proximo turno

                    enfermeiro.add_dia_trabalhado(dia, data_inicio_atividade - data_inicio_turno)
                    tecnico.add_dia_trabalhado(dia, data_inicio_atividade - data_inicio_turno)

                    # trocando de turno
                    enfermeiro = random.choice(enfermeiros_disponiveis)
                    enfermeiros_disponiveis.remove(enfermeiro)
                    if len(enfermeiros_disponiveis) == 0:
                        raise Exception('Enfermeiros insuficientes!')
                    tecnico = random.choice(tecnicos_disponiveis)
                    tecnicos_disponiveis.remove(tecnico)
                    if len(tecnicos_disponiveis) == 0:
                        raise Exception('Tecnicos insuficientes!')
                    fim_turno = fim_turno + datetime.timedelta(hours=horas_turno)
                    data_inicio_turno = data_inicio_atividade

                data_fim_atividade = data_inicio_atividade + datetime.timedelta(minutes=pontos_to_minutos(lista_nas[atividade]))

                if atividade in ['7a', '7b', '8a', '8b']:  # somente o enfermeiro
                    atendimento = Atendimento(paciente, data_inicio_atividade, data_fim_atividade, atividade, lista_nas[atividade],
                                              enfermeiro, None)
                else:  # enfermeiro, tecnico ou ambos
                    aux = random.randint(0, 3)
                    if aux == 0:  # somente o enfermeiro
                        atendimento = Atendimento(paciente, data_inicio_atividade, data_fim_atividade, atividade, lista_nas[atividade],
                                                  enfermeiro, None)
                    elif aux == 1:  # somente o tecnico
                        atendimento = Atendimento(paciente, data_inicio_atividade, data_fim_atividade, atividade, lista_nas[atividade],
                                                  None, tecnico)
                    else:  # enfermeiro e tecnico
                        data_fim_atividade = data_inicio_atividade + datetime.timedelta(minutes=pontos_to_minutos(lista_nas[atividade])/2)
                        atendimento = Atendimento(paciente, data_inicio_atividade, data_fim_atividade, atividade, lista_nas[atividade],
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
        diagnostico = random.choice(list(Diagnosticos.Index.keys()))
        paciente = Paciente(j+1, nome, diagnostico)
        pacientes.append(paciente)

    return pacientes


def pontos_to_minutos(pontos):
    return pontos * 14.4


def minutos_to_pontos(minutos):
    return minutos / 14.4


def exportar_horas_trabalhadas():

    with open('horas_trabalhadas.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Codigo', 'Tipo', 'Dia', 'Horas Trabalhadas'])

        for enfermeiro in enfermeiros:
            dias_trabalhados = enfermeiro.get_dias_horas_trabalhados()  # [dia, horas]
            for dia_trabalhado in dias_trabalhados:
                aux = [enfermeiro.get_codigo(), enfermeiro.get_tipo(), dia_trabalhado[0].strftime('%Y-%m-%d'),
                       dia_trabalhado[1]]
                filewriter.writerow(aux)
        for tecnico in tecnicos:
            dias_trabalhados = tecnico.get_dias_horas_trabalhados()  # [dia, horas]
            for dia_trabalhado in dias_trabalhados:
                aux = [tecnico.get_codigo(), tecnico.get_tipo(), dia_trabalhado[0].strftime('%Y-%m-%d'),
                       dia_trabalhado[1]]
                filewriter.writerow(aux)
    # TODO: ajustar para quando a qdade de horas trabalhadas por dia for maior q 24h
    return


def exportar_pacientes():
    with open('pacientes.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Codigo', 'Nome', 'Diagnostico'])

        for paciente in pacientes:
            aux = [paciente.get_codigo(), paciente.get_nome(), paciente.get_diagnostico()]
            filewriter.writerow(aux)


if __name__ == '__main__':

    #print(ProbabsNAS)
    pacientes = simular_pacientes(10)
    exportar_pacientes()
    data_inicio_sim = datetime.datetime(year=2022, month=1, day=1)
    total_dias = 5

    enfermeiros = simular_enfermeiros(7*10)
    tecnicos = simular_tecnicos(10*10)
    horas_turno = 12
    exportar_enfermeiros()

    atendimentos = simular_atendimentos()
    exportar_antendimentos(atendimentos)
    exportar_antendimentos_nn(atendimentos)
    exportar_horas_trabalhadas()
    # TODO: exportar relatorio de tempo por atividade por diagnositico & adicionar mais diagnosticos