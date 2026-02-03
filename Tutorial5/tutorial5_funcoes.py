"""
Funcoes para o Tutorial 5 de Processamento Digital de Sinais

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np


def PadraoRadiacao1(phi):
    """
    Sintetiza um padrão de radiação pre-definido sobre um vetor 'phi' de angulos.
    'phi' eh assumido estar em radianos.
    """

    k = np.array([-3, -2, -1, 0, 1, 2, 3])
    c = np.array([0.4, 0.15, 0.25, 0.5, 0.25, 0.15, 0.4])
    saida = 0.

    for n in range(k.shape[0]):
        saida += c[n]*np.exp(1j*k[n]*phi)

    return saida


def PadraoRadiacao2(phi):
    """
    Sintetiza um padrao de radiacao pre-definido sobre um vetor 'phi' de angulos.
    'phi' eh assumido estar em radianos.
    """

    k = np.array([-3, -2, -1, 0, 1, 2, 3])
    c = np.array([0.1*np.exp(-1j*np.pi), 0.3*np.exp(1j*np.pi/4),
                  0.2*np.exp(1j*np.pi/2), 0.0, 0.2*np.exp(-1j*np.pi/2),
                  0.3*np.exp(-1j*np.pi/4), 0.1*np.exp(+1j*np.pi)])
    saida = 0.

    for n in range(k.shape[0]):
        saida = saida + c[n]*np.exp(1j*k[n]*phi)

    return saida
