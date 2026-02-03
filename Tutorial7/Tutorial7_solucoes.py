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

# Onda dente-de-serra (serie de Fourier de senos)
coef_dente_sen = lambda m: -2*((-1)**(m+1))/(np.pi*m)

# onda quadrada (serie de Fourier de senos)
coef_quad_sen = lambda m: 2*(1-np.cos(np.pi*m))/(m*np.pi)

# onda triangular (serie de Fourier de senos)
coef_triang_cos = lambda m: 8*np.sin(m*np.pi/2)/ (np.pi*m)**2

# onda triangular (serie de Fourier de senos)
coef_triang_sen = lambda m: 2 * (np.cos(np.pi*m)-1)/(np.pi*m**2)


# %% cria coeficientes da serie de cosenos/senos

# numero de coeficientes a se usar na serie de Fourier de senos+cosenos
N = 25

assert N <= Nt, ("Numero de coeficientes 'N' eh maior que o numero de amostras no"
                 + " tempo 'Nt'- reduza o numero de coeficientes ou aumente a "
                 + "frequencia de amostragem!")


A_n = np.zeros(N)
B_n = np.zeros(N)

# sinal : ["arbitrario", "serra", "quad", "triang1", "triang2"]
sinal = "triang1"


match sinal:
    
    # coeficientes arbitrarios
    case "arbitrario":    
        A_n[1] = 1.2
        A_n[3] = 0.5

    # Onda dente de serra    
    case "serra":        
        B_n[1:] = coef_dente_sen(np.arange(1, N))
    
    # Onda quadrada
    case "quad":
        B_n[1:] = coef_quad_sen(np.arange(1, N))

    # Onda triangular (x entre [-1, 1])
    case "triang1":
        B_n[1:] = coef_triang_cos(np.arange(1, N))

    # Onda triangular (x entre 0 e +pi)
    case "triang2":
        A_n[0] = np.pi/2
        A_n[1:] = coef_triang_sen(np.arange(1, N))


# %% sintetiza sinal no tempo a partir dos coeficientes de Fourier
x = np.zeros(Nt)
for n in range(N):
    x += A_n[n]*np.cos(2*np.pi*n*f0*t) + B_n[n]*np.sin(2*np.pi*n*f0*t)

# calcula coeficientes da serie exponencial (DFT)
Xf_teorico = np.zeros(Nt, dtype='complex')
for n in range(N):
    
    # cos(x) = ( exp(1j*x) + exp(-1j*x) )/2
    Xf_teorico[n] += A_n[n]/2
    Xf_teorico[-n] += A_n[n]/2
       
    # sin(x) = ( exp(1j*x) - exp(-1j*x) ) / 2j
    Xf_teorico[n] += B_n[n]/2j
    Xf_teorico[-n] += -B_n[n]/2j

# --------------------------------------------------------------

plt.figure()
plt.plot(t, x)
plt.grid()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.title(f"Sinal periodico ({N} coeficientes)")

# %% calcula DFT do sinal

Ndft = x.shape[0]
df = fs/Ndft

Xf = np.fft.fft(x)/Ndft

# 'truque' para zerar valores numericamente muito proximos de zero
Xf[np.abs(Xf)<1e-15] = 0j

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

