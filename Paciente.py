
class Paciente:
    def __init__(self, codigo, nome, diagnostico):
        self.codigo = codigo
        self.nome = nome
        self.diagnostico = diagnostico

    def get_codigo(self):
        return self.codigo

    def get_nome(self):
        return self.nome

    def get_diagnostico(self):
        return self.diagnostico

    def __str__(self):
        return 'codigo=' + self.codigo + ' nome=' + self.nome + ', diagnostico=' + self.diagnostico

    def __repr__(self):
        return 'codigo=' + self.codigo + ' nome=' + self.nome + ', diagnostico=' + self.diagnostico
