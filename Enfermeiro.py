
class Enfermeiro:
    def __init__(self, cod_rfid, nome, tipo):
        self.cod_rfid = cod_rfid  # 8 digitos hexadecimais
        self.nome = nome
        self.tipo = tipo
        self.dias_trabalhados = []  # [dia, horas]
        self.dias_folgados = []  # [dia]

    def __str__(self):
        return 'cod_rfid=' + self.cod_rfid + ', nome=' + self.nome + ', tipo = ' + self.tipo

    def __repr__(self):
        return 'cod_rfid=' + self.cod_rfid + ', nome=' + self.nome + ', tipo = ' + self.tipo

    def get_codigo(self):
        return self.cod_rfid

    def get_nome(self):
        return self.nome

    def get_tipo(self):
        return self.tipo

    def get_dias_trabalhados(self):
        aux = []
        for i in self.dias_trabalhados:
            aux.append(i[0])
        return aux

    def get_dias_horas_trabalhados(self):
        return self.dias_trabalhados

    def add_dia_trabalhado(self, dia, horas):
        self.dias_trabalhados.append([dia, horas])
        return

    def get_dias_folgados(self):
        return self.dias_folgados

    def add_dia_folgado(self, dia):
        self.dias_folgados.append(dia)
