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
    Fev 2026
"""


import numpy as np

import matplotlib.pyplot as plt
plt.close("all")


# Frequencia de amostragem, em Hz
fs = 10000

# intervalo de amostragem no tempo, em segundos
dt = 1/fs

# frequencia fundamental [Hz]
f0 = 100.  

T = 1/f0                            # duracao [s]
Nt = int(T*fs)                      # No. de amostras no tempo
t = np.linspace(0, T-dt, Nt)        # vetor de amostras no tempo

# %% funcoes para obter coeficientes de Fourier 

def coef_dente_sen(k):
    # Onda dente-de-serra (serie de Fourier de senos)
    return -2*((-1)**(k+1))/(np.pi*k)

def coef_quad_sen(k):
    # onda quadrada (serie de Fourier de senos)
    return 2*(1-np.cos(np.pi*k))/(k*np.pi)

def coef_triang_sen(k):
    # onda triangular (serie de Fourier de senos)
    return 8*np.sin(k*np.pi/2)/ (np.pi*k)**2

def coef_triang_cos(k):
    # onda triangular (serie de Fourier de cossenos)
    return 2 * (np.cos(np.pi*k)-1)/(np.pi*k**2)


# %% cria coeficientes da serie de cosenos/senos

# numero de coeficientes a se usar na serie de Fourier de senos+cosenos
K = 25

assert K <= Nt, ("Kumero de coeficientes 'K' eh maior que o numero de amostras no"
                 + " tempo 'Nt'- reduza o numero de coeficientes ou aumente a "
                 + "frequencia de amostragem!")


A_k = np.zeros(K)
B_k = np.zeros(K)

# sinal : ["arbitrario", "serra", "quad", "triang1", "triang2"]
sinal = "triang2"


match sinal:
    
    # coeficientes arbitrarios
    case "arbitrario":    
        A_k[1] = 1.2
        A_k[3] = 0.5

    # Onda dente de serra    
    case "serra":        
        B_k[1:] = coef_dente_sen(np.arange(1, K))
    
    # Onda quadrada
    case "quad":
        B_k[1:] = coef_quad_sen(np.arange(1, K))

    # Onda triangular (x entre [-1, 1])
    case "triang1":
        B_k[1:] = coef_triang_sen(np.arange(1, K))

    # Onda triangular (x entre 0 e +pi)
    case "triang2":
        A_k[0] = np.pi/2
        A_k[1:] = coef_triang_cos(np.arange(1, K))


# %% sintetiza sinal no tempo a partir dos coeficientes de Fourier

x = np.zeros(Nt)
for n in range(K):
    x += A_k[n]*np.cos(2*np.pi*n*f0*t) + B_k[n]*np.sin(2*np.pi*n*f0*t)

# calcula coeficientes da serie exponencial (DFT)
Xf_teorico = np.zeros(Nt, dtype='complex')
for n in range(K):
    
    # cos(x) = ( exp(1j*x) + exp(-1j*x) )/2
    Xf_teorico[n] += A_k[n]/2
    Xf_teorico[-n] += A_k[n]/2
       
    # sin(x) = ( exp(1j*x) - exp(-1j*x) ) / 2j
    Xf_teorico[n] += B_k[n]/2j
    Xf_teorico[-n] += -B_k[n]/2j

# --------------------------------------------------------------

plt.figure()
plt.plot(t, x)
plt.grid()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.title(f"Sinal periodico ({K} coeficientes)")

# %% calcula DFT do sinal

Ndft = x.shape[0]
df = fs/Ndft

Xf = np.fft.fft(x)/Ndft

def zera_valores_pequenos(arr, tol=1e-15):
    """Zera valores reais ou imaginarios muito proximos de zero"""
    arr.real[np.abs(arr.real) < tol] = 0
    arr.imag[np.abs(arr.imag) < tol] = 0
    return arr

Xf = zera_valores_pequenos(Xf)
Xf_teorico = zera_valores_pequenos(Xf_teorico)


f = np.linspace(0, fs-df, Ndft)

plt.figure()

plt.subplot(211)
plt.plot(f, np.abs(Xf), ':s', label='DFT')
plt.plot(f, np.abs(Xf_teorico), '--o', label='Teorico')
plt.grid()
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(212)
plt.plot(f, np.angle(Xf), ':s')
plt.plot(f, np.angle(Xf_teorico), '--o')
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.ylabel("Fase [rad]")
plt.xlabel("Frequencia [Hz]")

