"""
Funcoes usadas no Tutorial 2 de Processamento Digital de Sinais e Aplicações em Acústica

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
import scipy.signal as ss

def sistema1(x):
    b, a = ss.iirpeak(0.25, 5, 1)
    y = ss.lfilter(b, a, x)
    n = np.arange(y.shape[0])
    return 1.5*np.cos(n*np.sqrt(2)/4)*y


def sistema2(x):
    return 2.5*np.tanh(0.74*x + 1.2)


def __criar_mat__():
    from scipy.io import savemat
    
    n = np.arange(50000)
    n1 = 10000
    n2 = 40000
    x1 = np.zeros(n.size)
    x1[n1:n2] = (np.sin(2*np.pi/500*n[n1:n2])
                       + 0.8*np.sin(2*np.pi/733*n[n1:n2]))
    x1[n1:n2] *= ss.windows.hann(n2-n1)
    
    x2 = (np.sin(2*np.pi/1250*n)
          *(1 + np.sin(2*np.pi/5000*n)))
    
    savemat('EnergiaPotencia.mat', {'x1': x1[:, np.newaxis],
                                    'x2': x2[:, np.newaxis]})