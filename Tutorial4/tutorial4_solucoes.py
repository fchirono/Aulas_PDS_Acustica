"""
Exercicio demonstrando o efeito de quantizacao de um sinal analogico durante
o processo de amostragem usando 'N' bits.

O exemplo tambem permite utilizar "dithering", conforme descrito em Lipshitz et
al (1992):
    
    "The object of dithering is to control the statistal properties of the total
    error and its relationship to the system input. In undithered systems we
    know that the error is a deterministic function of the input. If the input
    is simple or comparable in magnitude to the quantization step size, the
    total error signal is strongly input-dependent and audible as gross
    distortion and noise modulation. We shall see that use of dither with
    proper statistical properties can render the total error signal audibly
    equivalent to a steady white noise."
    
    S. Lipshitz et al, 1992


O processo exemplificado esta descrito nas seguintes referencias:
    
    - Secao 4.8.2 - Analog-to-Digital (A/D) Conversion
    A Oppenheim, R Schafer - Discrete-Time Signal Processing (2a Ed)
    Prentice-Hall, 1999
    
    - S. Lipshitz et al, "Quantization and Dither: A Theoretical Survey",
    J. Audio Eng. Soc., Vol. 40, No. 5, 1992.
    URL: https://secure.aes.org/forum/pubs/journal/?elib=7047
    https://www.researchgate.net/publication/236340550_Quantization_and_Dither_A_Theoretical_Survey


https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2025
"""

import time

import numpy as np
rng = np.random.default_rng()

import scipy.signal as ss

import matplotlib.pyplot as plt
plt.close('all')

import sounddevice as sd



def quantizar_sinal(sinal, n_bits, v_max=1.0):
    """
    Quantizar um sinal de tensao usando "Linear Pulse-Code Modulation" (L-PCM)
    com 'n_bits', e quantizador do tipo "midtread".
    
    Parametros:
    -----------
    sinal : array-like
        Sinal de tensao a ser quantizado
    
    n_bits : int
        Numero de bits utilizado na quantizacao
    
    v_max : float, opcional
        Tensao maxima de entrada do conversor AD (padrao: 1.0V)
    
    Retorna:
    --------
    Tupla contendo:
        - sinal_quantizado: sinal de tensao com amplitudes quantizadas
        - erro: sinal de erro (diferenca) entre sinal original e quantizado
        - tamanho_nivel: tamanho de cada nivel de quantizacao, em volts
    """
    
    # Converter argumento de entrada em array numpy
    sinal_tensao = np.array(sinal)
    
    # Calcular numero de niveis do conversor
    #   --> um bit eh utilizado para armazenar o sinal (+ ou -)
    n_niveis = 2**(n_bits-1)
    
    # tensao minima de entrada do conversor AD (bipolar)
    v_min = -v_max
    
    # Calcular tamanho de cada nivel binario
    tamanho_nivel = (v_max - v_min) / n_niveis
    
    # Criar array com os diferentes niveis de tensao eletrica
    # usando 'n_niveis' de '-v_max' ate (mas nao incluindo) '+v_max'
    niveis_tensao = np.linspace(v_min, v_max, n_niveis, endpoint=False)
    
    # Limitar o sinal para a faixa de tensao especificada
    #   --> O valor maximo deve ser um nivel abaixo de '+v_max'
    sinal_limitado = np.clip(sinal_tensao, v_min, v_max - tamanho_nivel)
    
    # Calcula o sinal quantizado usando um quantizador tipo "midtread"
    # (Lipshitz et al, 1992)
    sinal_quantizado = tamanho_nivel * np.floor(sinal_limitado/tamanho_nivel + 1/2)
    
    # Calcular o erro de quantizacao
    erro_quantizacao = sinal_tensao - sinal_quantizado
    
    return sinal_quantizado, erro_quantizacao, tamanho_nivel


# %% Gerar um sinal senoidal

# frequencia de amostragem [Hz]
fs = 44100

# intervalo de amostragem [s]
dt = 1/fs

# duracao do sinal [s]
T = 2.0

# vetor de amostras no tempo [s]
t = np.linspace(0, T-dt, int(T*fs))

# **************************************************
# Quantizar o sinal usando N bits (resultando em 2**(N_bits-1) niveis)
N_bits = 5
v_max = 5

# define o uso de dithering (True/False)
usar_dithering = False

# tensao do nivel menos significativo do conversor
v_lsb = (2*v_max)/(2**(N_bits-1))

# densidade espectral de tensao do ruido de conversao [V/sqrt(Hz)]
v_dig = v_lsb/np.sqrt(6*fs)

# **************************************************
# sinal senoidal de 4*LSB amplitude (pico-a-pico)

f0 = 123.4
sinal_original = (2*v_lsb) * np.sin(2 * np.pi * f0 * t)

if usar_dithering:    
    # dithering de espectro triangular, amplitude 2-LSB pico-a-pico
    #  --> soma de dois ruidos de espectro retangular independentes
    ruido1 = v_lsb*rng.uniform(low=-1, high=+1, size=t.shape[0])
    ruido2 = v_lsb*rng.uniform(low=-1, high=+1, size=t.shape[0])
    ruido = (ruido1 + ruido2)/2
    
    # # plotar histograma do ruido de dithering
    # plt.figure()
    # plt.hist(ruido, bins=100)
    
    sinal_original += ruido

sinal_quantizado, erro, tamanho_nivel = quantizar_sinal(sinal_original,
                                                        N_bits, v_max)
    
# %% plotar os primeiros 10 ms do sinal
plt.figure()
plt.subplot(211)
plt.plot(t[:int(0.01*fs)], sinal_original[:int(0.01*fs)])
plt.plot(t[:int(0.01*fs)], sinal_quantizado[:int(0.01*fs)], 'o-')
plt.grid()
plt.ylabel("Amplitude [V]")


plt.subplot(212)
plt.plot(t[:int(0.01*fs)], erro[:int(0.01*fs)])
plt.grid()
plt.ylabel("Amplitude [V]")
plt.xlabel("Tempo [s]")

# **************************************************
# plotar o espectro de amplitude do sinal 

f, espectro_original = ss.welch(sinal_original, fs, 'hann',
                                nperseg=2**12, noverlap=2**11,
                                scaling='density')

f, espectro_quantizado = ss.welch(sinal_quantizado, fs, 'hann',
                                nperseg=2**12, noverlap=2**11,
                                scaling='density')

f, espectro_erro = ss.welch(erro, fs, 'hann',
                            nperseg=2**12, noverlap=2**11,
                            scaling='density')

plt.figure()
plt.semilogy(f, espectro_original, label='Original')
plt.semilogy(f, espectro_quantizado, '--', label='Quantizado')
plt.semilogy(f, espectro_erro, ':', label='Erro')
plt.hlines(v_dig**2, f[0], f[-1], linestyles='--', color='k')
plt.legend()
plt.grid()
plt.ylabel("Magnitude")
plt.xlabel("Frequencia [Hz]")


# **************************************************
# auralizar o sinal original, seguido do sinal quantizado e da diferenca

# ***IMPORTANTE***: sempre atenue a amplitude do sinal antes de auralizar!

print("Auralizando o sinal original...")
sd.play(0.1*sinal_original, fs)

# 1 segundo de pausa entre os sinais
time.sleep(T+1)

print("Auralizando o sinal quantizado...")
sd.play(0.1*sinal_quantizado, fs)

time.sleep(T+1)

print("Auralizando a diferenca...")
sd.play(0.1*erro, fs)

# time.sleep(T+1)

# **************************************************

# Imprimir valores unicos no sinal quantizado para verificar quantidade
# correta
niveis_unicos = np.unique(sinal_quantizado)
print(f"Numero de bits: {N_bits}")
print(f"Numero de niveis de tensao: {len(niveis_unicos)}")
print(f"Niveis de quantizacao: {niveis_unicos}")
print(f"Tamanho do nivel: {tamanho_nivel:.6f}V")
print(f"Erro maximo de quantizacao: {np.max(np.abs(erro)):.6f}V")
