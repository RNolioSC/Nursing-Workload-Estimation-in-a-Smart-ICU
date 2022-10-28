from enum import Enum


class Paciente:
    def __init__(self, codigo, nome, diagnostico, data_admissao, data_alta):
        self.codigo = codigo
        self.nome = nome
        self.diagnostico = diagnostico
        self.data_admissao = data_admissao
        self.data_alta = data_alta  # patient discharge

    def get_codigo(self):
        return self.codigo

    def get_nome(self):
        return self.nome

    def get_diagnostico(self):
        return self.diagnostico

    def get_data_alta(self):
        return self.data_alta

    def __str__(self):
        return 'codigo=' + str(self.codigo) + ' nome=' + self.nome + ', diagnostico=' + self.diagnostico

    def __repr__(self):
        return 'codigo=' + str(self.codigo) + ' nome=' + self.nome + ', diagnostico=' + self.diagnostico
