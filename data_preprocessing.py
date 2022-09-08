import csv
from sklearn.preprocessing import normalize


def exportar_dados():
    with open('atendimentos_nn_prepr.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['Diagnostico', 'codPaciente', 'Duracao', 'Atividade', 'Pontuacao',
                             'Enfermeiro', 'Tecnico'])

        for i in data_norm:
            filewriter.writerow(i)


if __name__ == '__main__':
    with open("atendimentos_nn.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader, None)  # skip the headers
        tabela = [row for row in reader]

        diagnosticos = []
        codPacientes = []
        Duracoes = []
        Atividades = []
        Pontuacoes = []
        Enfermeiros = []
        Tecnicos = []

        # data preprocessing
        data_preprocessed = []
        for linha in tabela:
            nova_linha = []

            if linha[0] not in diagnosticos:
                diagnosticos.append(linha[0])
            nova_linha.append(diagnosticos.index(linha[0]))
            if linha[1] not in codPacientes:
                codPacientes.append(linha[1])
            nova_linha.append(codPacientes.index(linha[1]))
            if linha[2] not in Duracoes:
                Duracoes.append(linha[2])
            nova_linha.append(Duracoes.index(linha[2]))
            if linha[3] not in Atividades:
                Atividades.append(linha[3])
            nova_linha.append(Atividades.index(linha[3]))
            if linha[4] not in Pontuacoes:
                Pontuacoes.append(linha[4])
            nova_linha.append(Pontuacoes.index(linha[4]))
            if linha[5] not in Enfermeiros:
                Enfermeiros.append(linha[5])
            nova_linha.append(Enfermeiros.index(linha[5]))
            if linha[6] not in Tecnicos:
                Tecnicos.append(linha[6])
            nova_linha.append(Tecnicos.index(linha[6]))

            data_preprocessed.append(nova_linha)

        # data normalization
        data_norm = normalize(data_preprocessed, axis=0, norm='max')
        exportar_dados()
