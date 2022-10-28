import pickle
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
from nn_classification import evaluate_batch as nn_classif_evaluate_batch
from nn_regression import evaluate_batch as nn_regression_evaluate_batch
import numpy as np
from sklearn.metrics import mean_squared_error


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


def simular_nas(atividades, distribuicao='norm', variancia=1, lambd=1, scale=1, degr_freed=3):

    resultados = {}
    for j in atividades:
        if distribuicao == 'norm':
            media = PontosNAS[j]
            sigma = math.sqrt(variancia)
            aux = abs(stats.norm.rvs(media, sigma))
        elif distribuicao == 'exp':
            location = PontosNAS[j]
            scale = 1 / lambd
            aux = abs(stats.expon.rvs(loc=location, scale=scale))
        elif distribuicao == 't':
            location = PontosNAS[j]
            aux = abs(stats.t.rvs(df=degr_freed, loc=location, scale=scale))
        else:
            raise Exception('Distribuicao estatistica invalida')
        resultados[j] = aux
    return resultados


def exportar_antendimentos(atendimentos):
    print("exportar_antendimentos...")
    with open(simulacao_path + '/CSV/atendimentos.csv', 'w', newline='') as csvfile:
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
    with open(simulacao_path + '/CSV/atendimentos_nn.csv', 'w', newline='') as csvfile:
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


def simular_atendimentos(distribuicao):
    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio_sim + datetime.timedelta(i))

    pacientes_ativos = len(pacientes)
    atendimentos = []
    for dia in dias:
        print('Dia: ' + str(dia.date()))

        for _ in range(leitos - pacientes_ativos):  # leitos disponiveis
            print('leitos disponiveis=' + str(leitos - pacientes_ativos))
            if random.random() > 0.25:  # preencher leito
                nome = 'paciente' + str(pacientes[-1].get_codigo() + 1)
                diagnostico = random.choice(list(Diagnosticos.Index.keys()))
                [media_los, dp_los] = Diagnosticos.LOS[diagnostico]
                los_days = int(round(abs(stats.norm.rvs(media_los, dp_los))))
                data_alta = data_inicio_sim + datetime.timedelta(days=los_days)
                paciente = Paciente(pacientes[-1].get_codigo() + 1, nome, diagnostico, dia, data_alta)
                pacientes.append(paciente)
                pacientes_ativos += 1
                print('preenchido leito')

        for paciente in pacientes:
            if dia > paciente.data_alta:
                continue
            if dia.date() == paciente.data_alta.date():
                pacientes_ativos -= 1
                print('alta de paciente')

            atividades = escolher_atividades(paciente.get_diagnostico())
            lista_nas = simular_nas(atividades, distribuicao)

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
    with open(simulacao_path + '/CSV/enfermeiros.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Codigo RFID', 'Nome', 'Tipo'])

        for enfermeiro in enfermeiros:
            aux = [enfermeiro.get_codigo(), enfermeiro.get_nome(), enfermeiro.get_tipo()]

            filewriter.writerow(aux)
        for tecnico in tecnicos:
            aux = [tecnico.get_codigo(), tecnico.get_nome(), tecnico.get_tipo()]

            filewriter.writerow(aux)
    return


def simular_pacientes(quantidade, leitos, data_inicio_sim):
    if quantidade > leitos:
        raise Exception('Leitos insuficientes!')
    pacientes = []
    for j in range(quantidade):
        nome = 'paciente' + str(j + 1)
        diagnostico = random.choice(list(Diagnosticos.Index.keys()))
        [media_los, dp_los] = Diagnosticos.LOS[diagnostico]
        los_days = int(round(abs(stats.norm.rvs(media_los, dp_los))))
        data_alta = data_inicio_sim + datetime.timedelta(days=los_days)
        paciente = Paciente(j+1, nome, diagnostico, data_inicio_sim, data_alta)
        pacientes.append(paciente)

    return pacientes


def pontos_to_minutos(pontos):
    return pontos * 14.4


def minutos_to_pontos(minutos):
    return minutos / 14.4


def exportar_horas_trabalhadas():
    print("exportar_horas_trabalhadas...")
    with open(simulacao_path + '/CSV/horas_trabalhadas.csv', 'w', newline='') as csvfile:
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
    with open(simulacao_path + '/CSV/pacientes.csv', 'w', newline='') as csvfile:
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

    with open(simulacao_path + '/CSV/ativs_diag.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['codPaciente', 'Dia', 'Diagnostico', 'Atividades'])
        aux = ['', '', '']
        for i in range(1, 24):
            aux.append(str(i))
        filewriter.writerow(aux)
        for linha in tabela:
            filewriter.writerow(linha)


def salvar_result_teo(resultado_teorico):
    with open('resultados/teorico.bin', 'wb') as f:
        pickle.dump(resultado_teorico, f)


def salvar_result_sim(resultado_simulado):
    with open('resultados/simulado.bin', 'wb') as f:
        pickle.dump(resultado_simulado, f)


def salvar_result_nn_classif(resultado_nn_classif):
    with open('resultados/nn_classif.bin', 'wb') as f:
        pickle.dump(resultado_nn_classif, f)


def load_result_teo():
    with open('resultados/teorico.bin', 'rb') as f:
        return pickle.load(f)


def load_result_sim():
    with open('resultados/simulado.bin', 'rb') as f:
        return pickle.load(f)


def load_result_nn_classif():
    with open('resultados/nn_classif.bin', 'rb') as f:
        return pickle.load(f)


def salvar_result_nn_regression(resultado_nn_regression):
    with open('resultados/nn_regression.bin', 'wb') as f:
        pickle.dump(resultado_nn_regression, f)


def load_result_nn_regression():
    with open('resultados/nn_regression.bin', 'rb') as f:
        return pickle.load(f)


def calcular_resultados(recalcular_all=False, recalcular_teo=False, recalcular_sim=False, recalcular_nn_classif=False,
                        recalcular_nn_regression=False):
    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio_sim + datetime.timedelta(i))

    # resultados teoricos:
    if recalcular_teo or recalcular_all:
        resultado_teorico = []
        for dia in dias:
            t_parcial = 0.0
            for paciente in pacientes:
                if not (paciente.data_alta >= dia >= paciente.data_admissao):
                    continue
                probabs = Diagnosticos.Index[paciente.get_diagnostico()]
                for atividade in probabs:
                    t_parcial += probabs[atividade] * PontosNAS[atividade]
            resultado_teorico.append(t_parcial)
        salvar_result_teo(resultado_teorico)
    else:
        resultado_teorico = load_result_teo()

    # resultados simulado:
    if recalcular_sim or recalcular_all:
        resultado_simulado = []
        for dia in dias:
            t_parcial = 0.0
            for atendimento in atendimentos:
                if atendimento.get_diaHoraInicio().date() == dia.date():
                    t_parcial += float(atendimento.get_pontuacao_str())
            resultado_simulado.append(t_parcial)
        salvar_result_sim(resultado_simulado)
    else:
        resultado_simulado = load_result_sim()

    # resultados da rede neural de classificacao
    if recalcular_nn_classif or recalcular_all:
        resultado_nn_classif = []
        for dia in dias:
            t_parcial = 0.0

            if dia == datetime.datetime(year=2022, month=2, day=14):
                pass
            pacientes_do_dia = []
            for paciente in pacientes:
                if paciente.data_alta >= dia >= paciente.data_admissao:
                    pacientes_do_dia.append(paciente)
            diagnosticos = [p.get_diagnostico() for p in pacientes_do_dia]

            if len(pacientes_do_dia) < 2:
                pass
            all_atividades = nn_classif_evaluate_batch(diagnosticos, simulacao_path)
            print("nn_classif_evaluate: dia=", dia.date())

            for atividades in all_atividades:  # [1a,2,3,...]
                for atividade in atividades:
                    t_parcial += PontosNAS[atividade]
            resultado_nn_classif.append(t_parcial)
        salvar_result_nn_classif(resultado_nn_classif)
    else:
        resultado_nn_classif = load_result_nn_classif()

    # resultados da rede neural de regressao
    if recalcular_nn_regression or recalcular_all:
        resultado_nn_regression = []
        for dia in dias:
            t_parcial = 0.0
            pacientes_do_dia = []
            for paciente in pacientes:
                if paciente.data_alta >= dia >= paciente.data_admissao:
                    pacientes_do_dia.append(paciente)
            diagnosticos = [p.get_diagnostico() for p in pacientes_do_dia]
            all_atividades = nn_regression_evaluate_batch(diagnosticos, simulacao_path)
            print("nn_regress_evaluate: dia=", dia.date())

            for atividades in all_atividades:  # [duracao, ...]
                for atividade in atividades:
                    t_parcial += atividade
            resultado_nn_regression.append(t_parcial)
        salvar_result_nn_regression(resultado_nn_regression)
    else:
        resultado_nn_regression = load_result_nn_regression()

    plt.plot(resultado_teorico)
    plt.plot(resultado_simulado)
    plt.plot(resultado_nn_classif)
    plt.plot(resultado_nn_regression)
    plt.ylabel('Pontos NAS')
    plt.xlabel('Dia')
    plt.legend(['Resultado Teórico', 'Resultado Simulado', 'Resultado RN Classificação', 'Resultado RN Regressão'])
    plt.show()

    teo_vs_sim = [((y - x)*100)/x for x, y in zip(resultado_teorico, resultado_simulado)]
    teo_vs_teo = [((y - x) * 100) / x for x, y in zip(resultado_teorico, resultado_teorico)]
    teo_vs_nn_cl = [((y - x)*100)/x for x, y in zip(resultado_teorico, resultado_nn_classif)]
    teo_vs_nn_reg = [((y - x)*100)/x for x, y in zip(resultado_teorico, resultado_nn_regression)]
    plt.plot(teo_vs_teo)
    plt.plot(teo_vs_sim)
    plt.plot(teo_vs_nn_cl)
    plt.plot(teo_vs_nn_reg)
    plt.ylabel('Porcentagem')
    plt.xlabel('Dia')
    plt.legend(['Resultado Teórico', 'Resultado Simulado', 'Resultado RN Classificação', 'Resultado RN Regressão'])
    plt.show()

    # comparando com os resultados simulados
    # diff_teo_sim = [(x-y) for x, y in zip(resultado_teorico, resultado_simulado)]
    # diff_nn_reg_sim = [(x-y) for x, y in zip(resultado_nn_regression, resultado_simulado)]
    # plt.plot(diff_teo_sim)
    # plt.plot(diff_nn_reg_sim)
    # plt.ylabel('Diferenca (pontos NAS)')
    # plt.xlabel('Dia')
    # plt.legend(['Teorico menos simulado', 'Resultado NN Regressao menos simulado'])
    # plt.show()
    #
    # # em porcentagem comparado com resultado simulado
    # diff_teo_sim_perc = [((y - x) * 100) / x for x, y in zip(diff_teo_sim, resultado_simulado)]
    # diff_nn_reg_sim_perc = [((y - x) * 100) / x for x, y in zip(diff_nn_reg_sim, resultado_simulado)]
    # plt.plot(diff_teo_sim_perc)
    # plt.plot(diff_nn_reg_sim_perc)
    # plt.ylabel('Diferenca %')
    # plt.xlabel('Dia')
    # plt.legend(['Teorico menos simulado', 'Resultado NN Regressao menos simulado'])
    # plt.show()


def salvar_simulacao():
    with open(simulacao_path + '/bin/pacientes.bin', 'wb') as f:
        pickle.dump(pacientes, f)
    with open(simulacao_path + '/bin/data_inicio_sim.bin', 'wb') as f:
        pickle.dump(data_inicio_sim, f)
    with open(simulacao_path + '/bin/total_dias.bin', 'wb') as f:
        pickle.dump(total_dias, f)
    with open(simulacao_path + '/bin/enfermeiros.bin', 'wb') as f:
        pickle.dump(enfermeiros, f)
    with open(simulacao_path + '/bin/tecnicos.bin', 'wb') as f:
        pickle.dump(tecnicos, f)
    with open(simulacao_path + '/bin/horas_turno.bin', 'wb') as f:
        pickle.dump(horas_turno, f)
    with open(simulacao_path + '/bin/atendimentos.bin', 'wb') as f:
        pickle.dump(atendimentos, f)
    with open(simulacao_path + '/bin/diagnosticos_list.bin', 'wb') as f:
        pickle.dump(list(Diagnosticos.Index.keys()), f)


def load_pacientes(path):
    with open(path + '/bin/pacientes.bin', 'rb') as f:
        return pickle.load(f)


def load_data_inicio_sim(path):
    with open(path + '/bin/data_inicio_sim.bin', 'rb') as f:
        return pickle.load(f)


def load_total_dias(path):
    with open(path + '/bin/total_dias.bin', 'rb') as f:
        return pickle.load(f)


def load_enfermeiros(path):
    with open(path + '/bin/enfermeiros.bin', 'rb') as f:
        return pickle.load(f)


def load_tecnicos(path):
    with open(path + '/bin/tecnicos.bin', 'rb') as f:
        return pickle.load(f)


def load_horas_turno(path):
    with open(path + '/bin/horas_turno.bin', 'rb') as f:
        return pickle.load(f)


def load_atendimentos(path):
    with open(path + '/bin/atendimentos.bin', 'rb') as f:
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


def exportar_atendimentos(atendimentos):
    print('exportar_atendimentos...')
    to_export = [['codPaciente', 'diagnostico', 'atividade', 'pontosNAS']]
    for atendimento in atendimentos:
        aux = [atendimento.get_paciente().get_codigo(), atendimento.get_paciente().get_diagnostico(),
               atendimento.get_atividade_str(), atendimento.get_pontuacao_str()]
        to_export.append(aux)

    with open(simulacao_path + '/CSV/atendimentos.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for linha in to_export:
            filewriter.writerow(linha)


def exportar_duracao_ativs_por_diag():
    print("exportar_duracao_ativs_por_diag...")
    dias = []
    for i in range(0, total_dias):
        dias.append(data_inicio_sim + datetime.timedelta(i))

    tabela = []
    # ['codPaciente', 'Dia', 'Diagnostico', 'Duracao_atividades'])
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
                if len(atendimento.get_atividade_str()) > 1 and atendimento.get_atividade_str()[1] in 'abc':
                    linha[posAtiv + int(atendimento.get_atividade_str()[0])] = float(atendimento.get_pontuacao_str())
                else:
                    try:
                        linha[posAtiv + int(atendimento.get_atividade_str())] = float(atendimento.get_pontuacao_str())
                    except ValueError:
                        raise Exception("Erro em numero NAS de atendimento")

    with open(simulacao_path + '/CSV/duracao_ativs_diag.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['codPaciente', 'Dia', 'Diagnostico', 'Duracao_atividades'])
        aux = ['', '', '']
        for i in range(1, 24):
            aux.append(str(i))
        filewriter.writerow(aux)
        for linha in tabela:
            filewriter.writerow(linha)


def count_per_class(atendimentos):
    dados = []
    for atendimento in atendimentos:
        if atendimento.get_atividade_str() == '2':
            dados.append(float(atendimento.get_pontuacao_str()))

    dados = sorted(dados)
    classes = 75
    step = (max(dados) - min(dados)) / classes
    divider = min(dados) + step
    counter = 0
    dados_count = []
    scale = [divider]
    for dado in dados:
        if dado <= divider:
            counter += 1
        else:  # dado > dados
            counter = counter/len(dados)
            dados_count.append(counter*100)
            counter = 1
            scale.append(divider)
            divider += step
    scale.pop(0)
    return scale, dados_count, dados


def plot_distribuicao_dados(simulacao_path, path_outra_sim=None):

    scale, dados_count, dados = count_per_class(atendimentos)

    plt.plot(scale, dados_count)
    media = PontosNAS['2']

    count = 0
    for dado in dados:
        if media - 1 <= dado <= media + 1:
            count += 1
    print('porcentagem entre media + 1dp e -1 dp =', (count/len(dados))*100)

    if path_outra_sim is not None:
        atendimentos2 = load_atendimentos(path_outra_sim)
        scale2, dados_count2, _ = count_per_class(atendimentos2)
        plt.plot(scale2, dados_count2)
        plt.legend(['Dados de Validação', 'Dados de treinamento'])

    plt.ylabel('Porcentagem de atendimentos')
    plt.xlabel('Duracao (pontos NAS)')
    plt.title('Atividade 2: 4.3 pontos ~ 62 min')
    # plt.vlines([media-1, media+1], 0, max(dados_count), colors='k')
    # plt.vlines(media, 0, max(dados_count), colors='r')
    plt.show()


def comparacao_distr_estat():
    # x-axis ranges from -3 and 3 with .001 steps
    x = np.arange(-0, 10, 0.01)

    plt.plot(x, stats.norm.pdf(x, 5, 1)*100)  # (x, mean, std dev)
    # plt.plot(x, stats.expon.pdf(x, 5, 1)*100)  # (x, loc, scale)
    # plt.plot(x, stats.lognorm.pdf(x, 1, 5, 1)*100)  # (x, s, loc, scale)
    plt.plot(x, stats.t.pdf(x, 3, 5, 1)*100)  # (x, degr freed (v), loc, scale)

    plt.legend(['Normal', 'T Student'])
    # plt.legend(['Normal', 'Exponencial', 'Lognormal', 'T Student'])
    plt.ylabel('Porcentagem de atendimentos')
    plt.xlabel('Duracao (pontos NAS)')
    plt.show()

    # error_norm = [(0-b)**2 for b in data_norm]
    # error_expo = [(0-b)**2 for b in data_expon]
    # plt.plot(error_norm)
    # plt.plot(error_expo)
    # plt.show()


if __name__ == '__main__':

    nova_simulacao = True
    simulacao_path = 'simulacoes/simulacao3'
    if nova_simulacao:
        data_inicio_sim = datetime.datetime(year=2022, month=1, day=1)
        total_dias = 100
        leitos = 5
        pacientes = simular_pacientes(5, leitos, data_inicio_sim)
        enfermeiros = simular_enfermeiros(7*20)
        tecnicos = simular_tecnicos(10*20)
        horas_turno = 12
        atendimentos = simular_atendimentos(distribuicao='norm')
        salvar_simulacao()

    else:
        pacientes = load_pacientes(simulacao_path)
        data_inicio_sim = load_data_inicio_sim(simulacao_path)
        total_dias = load_total_dias(simulacao_path)
        enfermeiros = load_enfermeiros(simulacao_path)
        tecnicos = load_tecnicos(simulacao_path)
        horas_turno = load_horas_turno(simulacao_path)
        atendimentos = load_atendimentos(simulacao_path)
        Diagnosticos.add_atividades_faltantes()

    # exportar_ativs_por_diag()  # usado pra nn_classification
    # exportar_duracao_ativs_por_diag()  # usado pra nn_regression
    # plot_num_ativ_por_diag()
    plot_distribuicao_dados(simulacao_path, path_outra_sim='simulacoes/simulacao1')
    calcular_resultados(recalcular_all=True)
    comparacao_distr_estat()

    # exportar_enfermeiros()
    # exportar_pacientes()
    # exportar_antendimentos(atendimentos)
    # exportar_antendimentos_nn(atendimentos)
    # exportar_horas_trabalhadas()
    # TODO: CSV writting otimizar -- Dask
