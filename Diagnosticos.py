from ProbabsNas import ProbabsNAS
from pontosNAS import PontosNAS


def prob_rel(atividade, valor):
    # define um valor relativo, ie, um multiplicador
    return valor*ProbabsNAS[atividade]


def pts_rel(atividade, valor):
    # define um valor relativo, ie, um multiplicador
    return valor*PontosNAS[atividade]


Covid_probs = {
    # (BRUYNEEL et al., 2021)

    '1a': 220/905,
    '1b': 489/905,
    '1c': 196/905,
    '4a': 41/905,
    '4b': 535/905,
    '4c': 329/905,
    '6a': 53/905,
    '6b': 637/905,
    '6c': 215/905,

}
Covid_pts = {}

# caso o diagnostico nao seja conhecido, usar valores padrao
Desconhecido_probs = {}
Desconhecido_pts = {}

Index = {
    # [probablidade de cada atividade NAS ser escolhida, diferença de tempo empregado]

    'desconhecido': [Desconhecido_probs, Desconhecido_pts],
    'covid-19': [Covid_probs, Covid_pts]

}
