
class Atendimento:
    def __init__(self, paciente, data_inicio, data_fim,  atividade, pontuacao, enfermeiro, tecnico):
        self.paciente = paciente
        self.data_inicio = data_inicio
        self.data_fim = data_fim
        self.atividade = atividade
        self.pontuacao = pontuacao
        self.enfermeiro = enfermeiro
        self.tecnico = tecnico

    def __str__(self):
        return 'paciente=' + str(self.paciente) + ', data_inicio=' + str(self.data_inicio) + ', data_fim=' + \
               str(self.data_fim) + ', atividade=' + str(self.atividade) + ', pontuacao=' + str(self.pontuacao) + \
               ', enfermeiro=' + str(self.enfermeiro) + ', tecnico=' + str(self.tecnico)

    def __repr__(self):
        return 'paciente=' + str(self.paciente) + ', data_inicio=' + str(self.data_inicio) + ', data_fim=' + \
               str(self.data_fim) + ', atividade=' + str(self.atividade) + ', pontuacao=' + str(self.pontuacao) + \
               ', enfermeiro=' + str(self.enfermeiro) + ', tecnico=' + str(self.tecnico)

    def get_paciente_str(self):
        return str(self.paciente.get_nome())

    def get_dia_inicio_str(self):
        return self.data_inicio.strftime('%Y-%m-%d')

    def get_horario_inicio_str(self):
        return self.data_inicio.strftime('%H:%M:%S')

    def get_dia_fim_str(self):
        return self.data_fim.strftime('%Y-%m-%d')

    def get_horario_fim_str(self):
        return self.data_fim.strftime('%H:%M:%S')

    def get_atividade_str(self):
        return str(self.atividade)

    def get_pontuacao_str(self):
        return str(self.pontuacao)

    def get_enfermeiro(self):
        return self.enfermeiro

    def get_tecnico(self):
        return self.tecnico
