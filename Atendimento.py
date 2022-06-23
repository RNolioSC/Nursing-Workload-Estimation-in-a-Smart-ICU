
class Atendimento:
    def __init__(self, paciente, data, atividade, pontuacao, enfermeiro):  # TODO: adicionar tempo medio
        self.paciente = paciente
        self.data = data
        self.atividade = atividade
        self.pontuacao = pontuacao
        self.enfermeiro = enfermeiro

    def __str__(self):
        return 'paciente=' + str(self.paciente) + ', data=' + str(self.data) + ', atividade=' + str(self.atividade) +\
               ', pontuacao=' + str(self.pontuacao) + ', enfermeiro=' + str(self.enfermeiro)

    def __repr__(self):
        return 'paciente=' + str(self.paciente) + ', data=' + str(self.data) + ', atividade=' + str(self.atividade) +\
               ', pontuacao=' + str(self.pontuacao) + ', enfermeiro=' + str(self.enfermeiro)

    def get_paciente_str(self):
        return str(self.paciente)

    def get_dia_str(self):
        return self.data.strftime('%Y-%m-%d')

    def get_horario_str(self):
        return self.data.strftime('%H:%M:%S')

    def get_atividade_str(self):
        return str(self.atividade)

    def get_pontuacao_str(self):
        return str(self.pontuacao)

    def get_enfermeiro(self):
        return self.enfermeiro
