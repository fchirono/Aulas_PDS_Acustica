"""
Exercicio demonstrando o efeito de quantizacao de um sinal analogico durante
o processo de amostragem usando 'N' bits, com ou sem 'dithering'.

Referencias:
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
    Fev 2026
"""

import time

import numpy as np
rng = np.random.default_rng()

import scipy.signal as ss

import matplotlib.pyplot as plt
plt.close('all')

import sounddevice as sd

save_fig = False


def quantizar_sinal_dither(sinal, n_bits, v_max=1.0, dither=False):
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
    
    dither : bool, opcional
        Flag para aplicar dithering triangular ao sinal de entrada
    
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
    
    if dither:
        # dithering de espectro triangular, amplitude 2-LSB pico-a-pico
        #  --> soma de dois ruidos de espectro retangular independentes
        ruido1 = rng.uniform(low = -tamanho_nivel,
                             high = +tamanho_nivel,
                             size = t.shape[0])
        ruido2 = rng.uniform(low = -tamanho_nivel,
                             high = +tamanho_nivel,
                             size = t.shape[0])
        ruido = (ruido1 + ruido2)/2
        
        # adicionar o ruido ao sinal original
        sinal_tensao += ruido
        
        # # plotar histograma do ruido de dithering para confirmar
        # # a distribuicao triangular
        # plt.figure()
        # plt.hist(ruido, bins=100)
        
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
# sinal senoidal de 4*LSB amplitude (pico-a-pico)

f0 = 123.4

V0 = 1.25
# V0 = 10
# V0 = 0.5

sinal_original = V0 * np.sin(2 * np.pi * f0 * t)

# **************************************************
# Quantizar o sinal usando N bits (resultando em 2**(N_bits-1) niveis)
N_bits = 4
v_max = 5

# define o uso de dithering (True/False)
usar_dithering = True

# tensao do nivel menos significativo do conversor
v_lsb = (2*v_max)/(2**(N_bits-1))

# densidade espectral de tensao do ruido de conversao [V/sqrt(Hz)]
v_dig = v_lsb/np.sqrt(6*fs)

sinal_quantizado, erro, tamanho_nivel = quantizar_sinal_dither(sinal_original,
                                                               N_bits, v_max,
                                                               dither=usar_dithering)

# Imprimir algumas informacoes sobre o sinal quantizado
niveis_unicos = np.unique(sinal_quantizado)
print(f"Numero de bits: {N_bits}")
print(f"Niveis de quantizacao no sinal quantizado: {niveis_unicos}")
print(f"Tamanho do nivel: {tamanho_nivel:.6f}V")
print(f"Erro maximo de quantizacao: {np.max(np.abs(erro)):.6f}V")

    
# %% plotar os primeiros 10 ms do sinal

plt.figure()
plt.subplot(211)
plt.plot(t[:int(0.01*fs)], sinal_original[:int(0.01*fs)],
         label='x[n]')
plt.plot(t[:int(0.01*fs)], sinal_quantizado[:int(0.01*fs)], 'o-',
         label='x_Q[n]')
plt.grid()
plt.legend(loc="lower left")
plt.ylim([-2*V0, 2*V0])
plt.ylabel("Amplitude [V]")

plt.subplot(212)
plt.plot(t[:int(0.01*fs)], erro[:int(0.01*fs)], ':',
         color='C2', label='e[n]')
plt.grid()
plt.ylim([-2*V0, 2*V0])
plt.legend(loc="lower left")
plt.ylabel("Amplitude [V]")
plt.xlabel("Tempo [s]")

if save_fig:
    
    if usar_dithering:
        plt.savefig(f"2_1_SinalQuantizadoTempo_{V0}V_{N_bits}bits_dither.png")
    else:
        plt.savefig(f"2_1_SinalQuantizadoTempo_{V0}V_{N_bits}bits.png")    

# %% plotar o espectro de potencia do sinal 

Ndft = 1024

f, espectro_original = ss.welch(sinal_original, fs, 'hann',
                                nperseg=Ndft, noverlap=Ndft//2,
                                scaling='density')

f, espectro_quantizado = ss.welch(sinal_quantizado, fs, 'hann',
                                nperseg=Ndft, noverlap=Ndft//2,
                                scaling='density')

f, espectro_erro = ss.welch(erro, fs, 'hann',
                            nperseg=Ndft, noverlap=Ndft//2,
                            scaling='density')

plt.figure()
plt.semilogy(f, espectro_original, label='Original')
plt.semilogy(f, espectro_quantizado, '--', label='Quantizado')
plt.semilogy(f, espectro_erro, ':', label='Erro')
plt.hlines(v_dig**2, f[0], f[-1], linestyles='--', color='k')
plt.legend()
plt.grid()
# plt.xlim([0, 5e3])
plt.ylim([1e-10, 1e0])
plt.ylabel("Magnitude")
plt.xlabel("Frequencia [Hz]")

if save_fig:
    if usar_dithering:
        plt.savefig(f"2_1_SinalQuantizadoPSD_{V0}V_{N_bits}bits_dither.png")
    else:
        plt.savefig(f"2_1_SinalQuantizadoPSD_{V0}V_{N_bits}bits.png")
    
# %% auralizar o sinal original, seguido do sinal quantizado e da diferenca

# ***IMPORTANTE***: sempre atenue a amplitude do sinal antes de auralizar!

print("Auralizando o sinal original...")
sd.play(0.1*sinal_original, fs)

# 1 segundo de pausa entre os sinais
time.sleep(T+1)

print("Auralizando o sinal quantizado...")
sd.play(0.1*sinal_quantizado, fs)

time.sleep(T+1)

print("Auralizando o sinal de erro...")
sd.play(0.1*erro, fs)

# time.sleep(T+1)

