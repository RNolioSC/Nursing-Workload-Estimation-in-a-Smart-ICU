--CREATE DATABASE "bancodedados"

DROP TABLE IF EXISTS PacienteDiagnostico;
DROP TABLE IF EXISTS Diagnosticos;
DROP TABLE IF EXISTS AtendimentoProfEnf;
DROP TABLE IF EXISTS ProfissionaisEnf;
DROP TABLE IF EXISTS Atendimentos;
DROP TABLE IF EXISTS AtividadesNAS;
DROP TABLE IF EXISTS Pacientes;

CREATE TABLE Pacientes (codigo SERIAL PRIMARY KEY, nome VARCHAR(50)); 
INSERT INTO Pacientes(nome) VALUES ("Paciente1");

CREATE TABLE Diagnosticos(codigo SERIAL PRIMARY KEY, nome VARCHAR(50));

-- BIGINT UNSIGNED por compatibilidade com mariadb
CREATE TABLE PacienteDiagnostico(
	codPaciente BIGINT UNSIGNED, 
	codDiagnostico BIGINT UNSIGNED,
	PRIMARY KEY(codPaciente, codDiagnostico),
	CONSTRAINT fk1 FOREIGN KEY (codPaciente) REFERENCES Pacientes(codigo),
	CONSTRAINT fk2 FOREIGN KEY (codDiagnostico) REFERENCES Diagnosticos(codigo)
);

-- indice -> para sort by
CREATE TABLE AtividadesNAS(codigoNAS VARCHAR(2) PRIMARY KEY, descricao VARCHAR(500), pontos FLOAT, indice SERIAL);


CREATE TABLE Atendimentos (codigo SERIAL PRIMARY KEY, diaHora TIMESTAMP, 
						   atividade VARCHAR(2),
						   paciente BIGINT UNSIGNED,
						   FOREIGN KEY (atividade) REFERENCES AtividadesNAS(codigoNAS),
						   FOREIGN KEY (paciente) REFERENCES Pacientes(codigo)
						   );


CREATE TABLE ProfissionaisEnf (codigo VARCHAR(24) PRIMARY KEY, nome VARCHAR(50), tipo char);

CREATE TABLE AtendimentoProfEnf(
	codAtendimento BIGINT UNSIGNED,
	codProfEnf VARCHAR(24),
	PRIMARY KEY (codAtendimento, codProfEnf),
	FOREIGN KEY (codAtendimento) REFERENCES Atendimentos(codigo),
	FOREIGN KEY (codProfEnf) REFERENCES ProfissionaisEnf(codigo)
);
