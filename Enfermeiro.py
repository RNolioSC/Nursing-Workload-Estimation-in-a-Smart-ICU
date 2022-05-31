
class Enfermeiro:
    def __init__(self, cod_rfid, nome):
        self.cod_rfid = cod_rfid  # 8 digitos hexadecimais
        self.nome = nome

    def __str__(self):
        return 'cod_rfid=' + self.cod_rfid + ', nome=' + self.nome

    def __repr__(self):
        return 'cod_rfid=' + self.cod_rfid + ', nome=' + self.nome

    def get_codigo(self):
        return self.cod_rfid

    def get_nome(self):
        return self.nome
