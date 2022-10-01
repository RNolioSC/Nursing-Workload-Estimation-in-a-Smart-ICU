import pickle
import numpy as np

from matplotlib import pyplot as plt
import Diagnosticos
from pontosNAS import PontosNAS
import math
import scipy.stats as stats
import random
import datetime
from Atendimento import Atendimento
import csv
from Enfermeiro import Enfermeiro
from Paciente import Paciente
from nn_ativs_por_pac import evaluate as nn_evaluate
from nn_ativs_por_pac import evaluate_batch as nn_evaluate_batch


def escolher_atividades(diagnostico):
    probNasDiag = Diagnosticos.Index[diagnostico]
    Diagnosticos.add_atividades_faltantes()

    atividades = random.choices(['1a', '1b', '1c'],
                                weights=(probNasDiag['1a'], probNasDiag['1b'], probNasDiag['1c']))

    for i in ['2', '3']:
        if probNasDiag[i] >= random.random():
            atividades.append(i)

    atividades.extend(random.choices(['4a', '4b', '4c'],
                                     weights=(probNasDiag['4a'], probNasDiag['4b'], probNasDiag['4c'])))

    for i in ['5']:
        if probNasDiag[i] >= random.random():
            atividades.append(i)

    atividades.extend(random.choices(['6a', '6b', '6c'],
                                     weights=(probNasDiag['6a'], probNasDiag['6b'], probNasDiag['6c'])))
    atividades.extend(random.choices(['7a', '7b'],
                                     weights=(probNasDiag['7a'], probNasDiag['7b'])))
    atividades.extend(random.choices(['8a', '8b', '8c'],
                                     weights=(probNasDiag['8a'], probNasDiag['8b'], probNasDiag['8c'])))

    for i in ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']:
        if probNasDiag[i] >= random.random():
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
    print("exportar_antendimentos...")
    with open('CSV/atendimentos.csv', 'w', newline='') as csvfile:
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
    print("exportar_antendimentos_nn...")
    with open('CSV/atendimentos_nn.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Diagnostico', 'codPaciente', 'Duracao', 'Atividade', 'Pontuacao',
                             'Enfermeiro', 'Tecnico'])

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


def simular_atendimentos():
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
                if dia not in enfermeiro_aux.get_dias_trabalhados() and dia not in enfermeiro_aux.get_dias_folgados():
                    enfermeiros_disponiveis.append(enfermeiro_aux)
            if len(enfermeiros_disponiveis) == 0:
                raise Exception('Enfermeiros insuficientes!')

            tecnicos_disponiveis = []
            for tecnico_aux in tecnicos:
                if dia not in tecnico_aux.get_dias_trabalhados() and dia not in tecnico_aux.get_dias_folgados():
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
                    enfermeiro.add_dia_folgado(dia + datetime.timedelta(days=1))
                    tecnico.add_dia_trabalhado(dia, data_inicio_atividade - data_inicio_turno)
                    tecnico.add_dia_folgado(dia + datetime.timedelta(days=1))

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
    print("exportar_enfermeiros...")
    with open('CSV/enfermeiros.csv', 'w', newline='') as csvfile:
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
    print("exportar_horas_trabalhadas...")
    with open('CSV/horas_trabalhadas.csv', 'w', newline='') as csvfile:
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
    print("exportar_pacientes...")
    with open('CSV/pacientes.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Codigo', 'Nome', 'Diagnostico'])

        for paciente in pacientes:
            aux = [paciente.get_codigo(), paciente.get_nome(), paciente.get_diagnostico()]
            filewriter.writerow(aux)


def exportar_ativs_por_diag():
    print("exportar_ativs_por_diag...")
    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio_sim + datetime.timedelta(i))

    tabela = []
    # ['codPaciente', 'Dia', 'Diagnostico', 'Atividades'])
    # ['', '', '', '1', '2', '3', ...]
    for paciente in pacientes:
        for dia in dias:
            aux = [paciente.get_codigo(), dia.strftime('%Y-%m-%d'), paciente.get_diagnostico()]
            aux.extend([0] * 23)
            tabela.append(aux)

    posCodPac = 0
    posDia = 1
    posAtiv = 2
    for atendimento in atendimentos:
        for linha in tabela:
            if linha[posCodPac] == atendimento.get_paciente().get_codigo() and \
                    linha[posDia] == atendimento.get_dia_inicio_str():
                if len(atendimento.get_atividade_str()) > 1 and atendimento.get_atividade_str()[1] == 'a':
                    linha[posAtiv + int(atendimento.get_atividade_str()[0])] = 1
                elif len(atendimento.get_atividade_str()) > 1 and atendimento.get_atividade_str()[1] == 'b':
                    linha[posAtiv + int(atendimento.get_atividade_str()[0])] = 2
                elif len(atendimento.get_atividade_str()) > 1 and atendimento.get_atividade_str()[1] == 'c':
                    linha[posAtiv + int(atendimento.get_atividade_str()[0])] = 3
                else:
                    try:
                        linha[posAtiv + int(atendimento.get_atividade_str())] = 1
                    except ValueError:
                        raise Exception("Erro em numero NAS de atendimento")

    with open('CSV/ativs_diag.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['codPaciente', 'Dia', 'Diagnostico', 'Atividades'])
        aux = ['', '', '']
        for i in range(1, 24):
            aux.append(str(i))
        filewriter.writerow(aux)
        for linha in tabela:
            filewriter.writerow(linha)


def calcular_resultados():
    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio_sim + datetime.timedelta(i))

    # resultados teoricos:
    resultado_teorico = []
    for dia in dias:
        t_parcial = 0.0
        for paciente in pacientes:
            probabs = Diagnosticos.Index[paciente.get_diagnostico()]
            for atividade in probabs:
                t_parcial += probabs[atividade] * PontosNAS[atividade]
        resultado_teorico.append(t_parcial)

    # resultados praticos:
    resultado_pratico = []
    for dia in dias:
        t_parcial = 0.0
        for atendimento in atendimentos:
            if atendimento.get_diaHoraInicio().date() == dia.date():
                t_parcial += float(atendimento.get_pontuacao_str())
        resultado_pratico.append(t_parcial)

    # resultados da rede neural
    # TODO: mudar para evaluar por batch
    # TODO: pode ser amostral.
    #
    # resultado_nn = []
    # for dia in dias:
    #     t_parcial = 0.0
    #     for paciente in pacientes:
    #         atividades = nn_evaluate(paciente.get_diagnostico())
    #         print('evaluate paciente='+str(paciente.get_codigo()) + ', dia='+str(dia.date()))
    #         for atividade in atividades:
    #             t_parcial += PontosNAS[atividade]
    #     resultado_nn.append(t_parcial)

    # nn por batch:
    resultado_nn = []
    for dia in dias:
        t_parcial = 0.0
        diagnosticos = [p.get_diagnostico() for p in pacientes]
        all_atividades = nn_evaluate_batch(diagnosticos)
        print("dia=", dia.date())

        for atividades in all_atividades:  # [1a,2,3,...]
            for atividade in atividades:
                t_parcial += PontosNAS[atividade]
        resultado_nn.append(t_parcial)

        # print('evaluate dia=' + str(dia.date()))
        # for paciente in pacientes:
        #     atividades = nn_evaluate(paciente.get_diagnostico())
        #     #print('evaluate paciente=' + str(paciente.get_codigo()) + ', dia=' + str(dia.date()))
        #     for atividade in atividades:
        #         t_parcial += PontosNAS[atividade]
        #resultado_nn.append(t_parcial)

    # TODO: adicionar a porcentagem no primeiro grafico.

    plt.plot(resultado_teorico)
    plt.plot(resultado_pratico)
    plt.plot(resultado_nn)
    plt.ylabel('Pontos NAS')
    plt.xlabel('Dia')
    plt.legend(['Resultado Teorico', 'Resultado Prático', 'Resultado Rede Neural'])
    plt.show()

    teo_vs_pr = [((y - x)*100)/x for x, y in zip(resultado_teorico, resultado_pratico)]
    teo_vs_teo = [((y - x) * 100) / x for x, y in zip(resultado_teorico, resultado_teorico)]
    plt.plot(teo_vs_teo)
    plt.plot(teo_vs_pr)
    plt.ylabel('Porcentagem')
    plt.xlabel('Dia')
    plt.legend(['Resultado Teorico', 'Resultado Prático'])
    plt.show()

    # plt.plot(debug_teo[0:100])
    # plt.plot(debug_pr[0:100])
    # plt.ylabel('Pontos NAS')
    # plt.xlabel('Dia')
    # plt.legend(['Resultado Teorico', 'Resultado Prático'])
    # plt.show()

    # aten_teo = []
    # aten_pra = []
    # for atendimento in atendimentos:
    #     aten_pra.append(float(atendimento.get_pontuacao_str()))
    #     aten_teo.append(PontosNAS[atendimento.get_atividade_str()])
    # plt.plot(aten_teo[0:100])
    # plt.plot(aten_pra[0:100])
    # plt.ylabel('Pontos NAS')
    # plt.xlabel('Atendimento')
    # plt.legend(['Resultado Teorico', 'Resultado Prático'])
    # plt.show()


def salvar_simulacao():
    with open('bin/pacientes.bin', 'wb') as f:
        pickle.dump(pacientes, f)
    with open('bin/data_inicio_sim.bin', 'wb') as f:
        pickle.dump(data_inicio_sim, f)
    with open('bin/total_dias.bin', 'wb') as f:
        pickle.dump(total_dias, f)
    with open('bin/enfermeiros.bin', 'wb') as f:
        pickle.dump(enfermeiros, f)
    with open('bin/tecnicos.bin', 'wb') as f:
        pickle.dump(tecnicos, f)
    with open('bin/horas_turno.bin', 'wb') as f:
        pickle.dump(horas_turno, f)
    with open('bin/atendimentos.bin', 'wb') as f:
        pickle.dump(atendimentos, f)
    with open('bin/diagnosticos_list.bin', 'wb') as f:
        pickle.dump(list(Diagnosticos.Index.keys()), f)


def load_pacientes():
    with open('bin/pacientes.bin', 'rb') as f:
        return pickle.load(f)


def load_data_inicio_sim():
    with open('bin/data_inicio_sim.bin', 'rb') as f:
        return pickle.load(f)


def load_total_dias():
    with open('bin/total_dias.bin', 'rb') as f:
        return pickle.load(f)


def load_enfermeiros():
    with open('bin/enfermeiros.bin', 'rb') as f:
        return pickle.load(f)


def load_tecnicos():
    with open('bin/tecnicos.bin', 'rb') as f:
        return pickle.load(f)


def load_horas_turno():
    with open('bin/horas_turno.bin', 'rb') as f:
        return pickle.load(f)


def load_atendimentos():
    with open('bin/atendimentos.bin', 'rb') as f:
        return pickle.load(f)


def plot_num_ativ_por_diag():
    n_ativs_desconhecido = {}
    for i in PontosNAS:
        n_ativs_desconhecido[i] = 0
    n_ativs_covid = n_ativs_desconhecido.copy()
    n_ativs_queimado = n_ativs_desconhecido.copy()
    n_ativs_trauma = n_ativs_desconhecido.copy()
    n_pacientes_por_diag = [0, 0, 0, 0]

    for atendimento in atendimentos:
        if atendimento.get_paciente().get_diagnostico() == 'desconhecido':
            n_ativs_desconhecido[atendimento.get_atividade_str()] += 1
        elif atendimento.get_paciente().get_diagnostico() == 'covid-19':
            n_ativs_covid[atendimento.get_atividade_str()] += 1
        elif atendimento.get_paciente().get_diagnostico() == 'queimado':
            n_ativs_queimado[atendimento.get_atividade_str()] += 1
        else:
            n_ativs_trauma[atendimento.get_atividade_str()] += 1
    for paciente in pacientes:
        if paciente.get_diagnostico() == 'desconhecido':
            n_pacientes_por_diag[0] += 1
        elif paciente.get_diagnostico() == 'covid-19':
            n_pacientes_por_diag[1] += 1
        elif paciente.get_diagnostico() == 'queimado':
            n_pacientes_por_diag[2] += 1
        else:
            n_pacientes_por_diag[3] += 1

    plt.scatter(list(PontosNAS.keys()), [a / (n_pacientes_por_diag[0] * total_dias) for a in list(n_ativs_desconhecido.values())], marker='o', zorder=2)
    plt.scatter(list(PontosNAS.keys()), [a / (n_pacientes_por_diag[1] * total_dias) for a in list(n_ativs_covid.values())], marker='s', zorder=3)
    plt.scatter(list(PontosNAS.keys()), [a / (n_pacientes_por_diag[2] * total_dias) for a in list(n_ativs_queimado.values())], marker='x', zorder=4)
    plt.scatter(list(PontosNAS.keys()), [a / (n_pacientes_por_diag[3] * total_dias) for a in list(n_ativs_trauma.values())], marker='^', zorder=5)
    plt.xlabel('Atividades')
    plt.ylabel('Frequencia média por dia por paciente')
    plt.legend(['Desconhecido', 'Covid-19', 'Queimado', 'Trauma'])

    plt.grid(axis='x', zorder=1)
    plt.show()
    # TODO: add outro eixo com o num total de amostras, ie: list(n_ativs_trauma.values())


if __name__ == '__main__':

    nova_simulacao = False
    if nova_simulacao:
        pacientes = simular_pacientes(50)
        data_inicio_sim = datetime.datetime(year=2022, month=1, day=1)
        total_dias = 365
        enfermeiros = simular_enfermeiros(7*50)
        tecnicos = simular_tecnicos(10*50)
        horas_turno = 12
        atendimentos = simular_atendimentos()
        salvar_simulacao()

    else:
        pacientes = load_pacientes()
        data_inicio_sim = load_data_inicio_sim()
        total_dias = load_total_dias()
        enfermeiros = load_enfermeiros()
        tecnicos = load_tecnicos()
        horas_turno = load_horas_turno()
        atendimentos = load_atendimentos()
        Diagnosticos.add_atividades_faltantes()

    #exportar_ativs_por_diag()
    #plot_num_ativ_por_diag()
    calcular_resultados()

    # exportar_enfermeiros()
    # exportar_pacientes()
    # exportar_antendimentos(atendimentos)
    # exportar_antendimentos_nn(atendimentos)
    # exportar_horas_trabalhadas()
    # TODO: CSV writting otimizar -- Dask
