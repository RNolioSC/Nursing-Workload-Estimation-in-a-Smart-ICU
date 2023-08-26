
import serial
import mysql.connector
import argparse
import mariadb
from datetime import datetime
import time
import sys

def novo_atendimento(user, pw, tag, cod_paciente=1):
        try:
                conn = mariadb.connect(
                user=user,
                password=pw,
                host="localhost",
                port=3306,
                database="bancodedados"
                )

        except mariadb.Error as e:
                print(e)
                sys.exit(1)
                
        cur = conn.cursor(buffered=True)  # run multiple querries
        try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cur.execute("INSERT INTO Atendimentos (diaHora, atividade, paciente) VALUES (?, ?, ?) RETURNING codigo", 
                (timestamp, "1a", cod_paciente))
                conn.commit()
                cod_atendimento = cur.fetchone()[0]
                cur.nextset()
                cur.execute("INSERT INTO AtendimentoProfEnf (codAtendimento, codProfEnf) VALUES (?, ?)", 
                (cod_atendimento, tag))
                conn.commit()
                
                print("atendimento registrado: " + tag + ", " + timestamp)
        except mariadb.Error as e:
                print(e)
        
def cadastrar_tag(user, pw, tag, nome, tipo):
        try:
                conn = mariadb.connect(
                user=user,
                password=pw,
                host="localhost",
                port=3306,
                database="bancodedados"
                )

        except mariadb.Error as e:
                print(e)
                sys.exit(1)
                
        cur = conn.cursor()
        try:
                cur.execute("INSERT INTO ProfissionaisEnf (codigo, nome, tipo) values (?, ?, ?);", (tag, nome, tipo[0]))
                conn.commit()
                print("cadastrado: " + tag + ", " + nome)
        except mariadb.Error as e:
                print(e)

if __name__ == "__main__":

        ser = serial.Serial('/dev/ttyUSB0')
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--cadastrar", action='store_true', help="cadastrar nova tag")
        parser.add_argument("-n", "--nome", type=str, help="nome do profissional de enfermagem")
        parser.add_argument("-t", "--tipo", type=str, help="e=enfermeiro ou t=tecnico")
        #parser.add_argument("-u", "--username", type=str,  required=True, help="mariadb username")
        #parser.add_argument("-p", "--password", type=str, required=True, help="mariadb password")
        args = parser.parse_args()
        cad = args.cadastrar
        nome = args.nome
        tipo = args.tipo
        #user = args.username
        user = "admin"
        #pw = args.password
        pw = "admin"

        while True:
            if ser.in_waiting:
                bs = ser.read(ser.in_waiting)
                print("bs.len="+str(len(bs)))
                bs = '/'.join([f'0x{b:x}' for b in bs])
                #print("bs="+bs)
                bs = bs.removeprefix('0xcc/0xff/0xff/0x10/0x32/0xd/0x1/')
                lastchar = bs[-1]
                while lastchar != '/':
                        bs = bs[:-1]
                        lastchar = bs[-1]
                bs = bs[:-1] # remove '/'
                #print("bs: " + bs)
                bs = bs.removeprefix('0x')
                temp = bs.split('/0x')
                tag = ''
                for i in temp:
                        if len(i)<2:
                                tag+='0'
                        tag += i
                print(tag)
                if cad == True:
                        cadastrar_tag(user, pw, tag, nome, tipo)
                        break
                else:
                        novo_atendimento(user, pw, tag)
                

