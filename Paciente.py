
class Paciente:
    def __init__(self, codigo, nome):
        self.codigo = codigo
        self.nome = nome

    def get_codigo(self):
        return self.codigo

    def get_nome(self):
        return self.nome

    def __str__(self):
        return 'codigo=' + self.codigo + ' nome=' + self.nome

    def __repr__(self):
        return 'codigo=' + self.codigo + ' nome=' + self.nome
