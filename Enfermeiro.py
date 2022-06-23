
class Enfermeiro:
    def __init__(self, cod_rfid, nome, tipo):
        self.cod_rfid = cod_rfid  # 8 digitos hexadecimais
        self.nome = nome
        self.tipo = tipo

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
