# -*- coding: utf-8 -*-
"""
Funcoes e script para calcular os coeficientes da Serie de Fourier de um sinal
atraves da Transformada Discreta de Fourier (DFT).

Para obter os coeficientes da Serie de Fourier diretamente da Transformada
Discreta de Fourier (DFT/FFT), eh necessario analisar exatamente um periodo
inteiro do sinal de interesse:
    - o termo 0 da DFT ira informar o valor da componente DC do sinal;
    - o termo 1 da DFT ira entregar o valor do primeiro harmonico do sinal;
    - o termo 2 da DFT ira entregar o valor do segundo harmonico do sinal;
    - etc.

A Serie de Fourier eh aqui definida na forma:
    
    f(t)    =  sum(n=0 to N-1)     [A[n] * sin(2 * pi * n*f0 * t)]
               + sum(n=0 to N-1)    [B[n] * cos(2 * pi * n*f0 * t)] 
         
            = sum(n=-inf to +inf)   [C[n] * exp(-1j * n * (2*pi*f0) * t)]

onde A[n] e B[n] sao vetores de coeficientes reais da serie de seno e coseno,
e C[n] eh um vetor de coeficientes complexos da serie exponencial.


https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Ago 2026
"""


import numpy as np

import matplotlib.pyplot as plt
plt.close("all")


def zera_valores_pequenos(arr, tol=1e-13):
    """Zera valores reais ou imaginarios muito proximos de zero"""
    arr.real[np.abs(arr.real) < tol] = 0
    arr.imag[np.abs(arr.imag) < tol] = 0
    return arr


fs = 10000                          # Frequencia de amostragem [Hz]
dt = 1/fs                           # intervalo de amostragem [s]

f0 = 100.                           # frequencia fundamental [Hz]
T = 1./f0                           # duracao [s]
Nt = int(T*fs)                      # No. de amostras no tempo
t = np.linspace(0, T-dt, Nt)        # vetor de amostras no tempo

K = 25                              # numero de coeficientes da Serie de Fourier



# %% cria coeficientes da serie de cosenos/senos

assert (K-1)*f0 < fs/2, ("Condicao de Nyquist nao esta obedecida! Reduza o numero"
                          + " de coeficientes ou aumente a frequencia de amostragem!")


A_k = np.zeros(K)
B_k = np.zeros(K)

# sinal : ["serra", "triang"]
sinal = "triang"

match sinal:

    # Onda dente de serra    
    case "serra":        
        k = np.arange(1, K)
        B_k[1:] = -2*np.cos(np.pi*k)/(np.pi*k)
    
    # Onda triangular (funcao par)
    case "triang":
        k =np.arange(1, K)
        A_k[1:] = 8 * np.sin(k*np.pi/2)**2 / (np.pi*k)**2


# %% sintetiza sinal no tempo a partir dos coeficientes de Fourier

x = np.zeros(Nt)
for n in range(K):
    x += A_k[n]*np.cos(2*np.pi*n*f0*t) + B_k[n]*np.sin(2*np.pi*n*f0*t)

# calcula coeficientes da serie exponencial
C_k = np.zeros(Nt, dtype='complex')
for n in range(K):
    C_k[n] += A_k[n]/2 - 1j*B_k[n]/2
    C_k[-n] += A_k[n]/2 + 1j*B_k[n]/2
       

# --------------------------------------------------------------

plt.figure()
plt.plot(t, x)
plt.grid()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.title(f"Sinal periodico ({K} coeficientes)")
plt.ylim([-1.2, 1.2])

# %% calcula DFT do sinal

df = fs/Nt

Xf = np.fft.fft(x)

Xf = zera_valores_pequenos(Xf)
C_k = zera_valores_pequenos(C_k)

f = np.linspace(0, fs-df, Nt)

plt.figure(figsize=(12, 8))

plt.subplot(211)
plt.plot(f, np.abs(Xf/Nt), ':s', label='X[k]/Nt')
plt.plot(f, np.abs(C_k), '--o', label='Ck')
plt.grid()
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(212)
plt.plot(f, np.angle(Xf/Nt), ':s')
plt.plot(f, np.angle(C_k), '--o')
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.ylabel("Fase [rad]")
plt.xlabel("Frequencia [Hz]")

