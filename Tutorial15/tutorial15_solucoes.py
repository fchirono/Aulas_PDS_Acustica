# -*- coding: utf-8 -*-
"""
Funcoes e script para criar sinais modulados AM e FM, para usar em exercicio
com transformada de Hilbert.

Este script cria um sinal aleatorio de banda limitada, e usa este sinal para
modular uma portadora senoidal de 10 kHz em AM e FM. As portadoras possuem
amplitude unitaria (1 V pico).

Sugestao de exercicio:
    - Entregar os sinais AM e FM em um arquivo (e.g. .mat), junto das
    propriedades dos sinais (freq. e amplitude da portadora, sensibilidade em
    frequencia, etc)
    - Alunos implementam a funcao de transformada de Hilbert
    - Alunos realizam a demodulacao dos sinais, e comparam os sinais moduladores

Funcoes de criar sinais AM/FM copiadas do pacote MOSQITO: https://github.com/Eomys/MoSQITo

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np

import scipy.signal as ss

import matplotlib.pyplot as plt
plt.close("all")

import sounddevice as sd


#%% funcoes para criar sinais AM/FM - adaptadas de https://github.com/Eomys/MoSQITo

def gerador_seno_am(xmod, fs, fc, print_m=False):
    """ Geração de onda senoidal modulada em amplitude (AM)
    
    Esta função cria um sinal modulado em amplitude (AM) com portadora 
    senoidal de frequência 'fc', e sinal modulante arbitrário 'xmod'.
    O comprimento do sinal AM é igual ao comprimento de 'xmod'. 
    O sinal portador tem amplitude de pico unitária.

    Parâmetros
    ----------
    xmod: array
        Sinal modulante, dim(N).
    
    fs: float
        Frequência de amostragem, em Hz.
    
    fc: float
        Frequência da portadora, em Hz. Deve ser menor que 'fs/2'.
    
    print_m: bool, opcional
        Flag que indica se o índice de modulação calculado deve ser impresso.
        Padrão é False.
    
    Retorna
    -------
    y: numpy.array
        Sinal modulado em amplitude com portadora senoidal, em Pascals, dim(N).
    m: float
        Índice de modulação    
        
    Aviso
    -----
    spl_level deve ser fornecido em dB, ref=2e-5 Pa.
        
    Notas
    -----
    O índice de modulação 'm' será igual ao valor de pico do sinal 
    modulante 'xmod'. Seu valor pode ser impresso definindo a flag 
    opcional 'print_m' como True.
    
    Para 'm' = 0,5, a amplitude da portadora varia 50% acima e abaixo do 
    seu nível não modulado. Para 'm' = 1,0, ela varia 100%. Com 100% de 
    modulação, a amplitude da onda às vezes chega a zero, o que representa 
    modulação total. Aumentar o sinal modulante além desse ponto é 
    conhecido como sobremodulação.
    """
    
    assert fc < fs/2, "A frequência da portadora 'fc' deve ser menor que 'fs/2'!"
    
    Nt = xmod.shape[0]        # comprimento do sinal em amostras
    T = Nt/fs               # comprimento do sinal em segundos
    dt = 1/fs               # intervalo de amostragem em segundos

    # vetor de amostras temporais
    t = np.linspace(0, T-dt, int(T*fs))
    
    # portadora senoidal de amplitude unitária com frequência 'fc' [Hz]
    xc = np.sin(2*np.pi*fc*t)

    # sinal AM
    y_am = (1 + xmod)*xc

    # índice de modulação
    m = np.max(np.abs(xmod))

    if print_m:
        print(f"Índice de modulação AM = {m}")
    
    if m > 1:
        print("Aviso ['gerador_seno_am']: índice de modulação m > 1\n\tO sinal está sobremodulado!")

    return y_am, m


def gerador_seno_fm(xmod, fs, fc, k, print_info=False):
    """
    Cria um sinal modulado em frequência (FM) de nível 'spl_level' (em dB SPL)
    com portadora senoidal de frequência 'fc', sinal modulante arbitrário
    'xm', sensibilidade de frequência 'k', e frequência de amostragem 'fs'. 
    O comprimento do sinal FM é igual ao comprimento de 'xm'. 
    
    Parâmetros
    ----------
    xmod: array
        Sinal modulante, dim(N)
    fs: float
        Frequência de amostragem, em [Hz].
    fc: float
        Frequência da portadora, em [Hz]. Deve ser menor que 'fs/2'.
    k: float
        Sensibilidade de frequência do modulador. 
    print_info: bool, opcional
        Se True, o desvio máximo de frequência e o índice de modulação 
        são impressos. Padrão é False
    
    Retorna
    -------
    y_fm: numpy.array
        Sinal modulado em frequência com portadora senoidal, dim(N) em [Pa].
    inst_freq: numpy.array
        Frequência instantânea, dim(N)
    max_freq_deviation: float
        Desvio máximo de frequência [Hz]   
    FM_modulation_index: float
        Índice de modulação 
        
    
    Notas
    -----
    A sensibilidade de frequência 'k' é igual ao desvio de frequência em Hz 
    em relação a 'fc' por unidade de amplitude do sinal modulante 'xmod'.
           
    """
    
    assert fc < fs/2, "A frequência da portadora 'fc' deve ser menor que 'fs/2'!"
    
     # intervalo de amostragem em segundos
    dt = 1/fs

    # frequência instantânea do sinal FM
    inst_freq = fc + k*xmod
    
    # sinal FM de amplitude unitária
    y_fm = np.sin(2*np.pi * np.cumsum(inst_freq)*dt)
    
    # desvio máximo de frequência
    f_delta = k * np.max(np.abs(xmod))
    
    # índice de modulação FM
    m = np.max(np.abs(2*np.pi * k * np.cumsum(xmod)*dt))

    if print_info:
        print(f'\tDesvio máximo de frequência: {f_delta} Hz')
        print(f'\tÍndice de modulação FM: {m:.2f}')

    return y_fm, inst_freq, f_delta, m


#%% transformada de Hilbert

def calc_sinal_analitico(x):
    """
    Retorna o sinal analitico dado por 'x + 1j*y', onde 'y' eh a 
    Transformada de Hilbert do sinal 'x'.
    """
    
    N = x.shape[0]
    xf = np.fft.fft(x)
    
    # cria vetor de valores para multiplicar o espectro de 'x'
    h = np.zeros(N)
    
    # frequencia zero e Nyquist sao multiplicadas por um
    h[0] = h[N // 2] = 1
    
    # todas as outras frequencias positivas sao multiplicadas por dois
    h[1:N // 2] = 2
    
    return np.fft.ifft(xf*h)


# %% cria vetor temporal

fs = 48000
dt = 1/fs

T = 5      # [s]

Nt = int(fs*T)-1

t = np.linspace(0, T-dt, Nt)


# %%  criar sinal modulador (passa-baixas)

gerador = np.random.default_rng()
ruidobranco = gerador.normal(loc=0.0, scale=1.0, size=Nt)

# filtro tipo Butterworth de 4a ordem, freq de corte 3 Hz
Nfiltro = 4
f_corte = 3
filtro = ss.butter(Nfiltro, f_corte, btype='low', output='sos', fs=fs)

# filtra ruido branco para obter ruido de banda limitada (passa-baixas)
ruido_pb = ss.sosfilt(filtro, ruidobranco)

# normaliza sinal passa-baixas para amplitude maxima de 0.5
ruido_pb *= 0.5/np.max(np.abs(ruido_pb))

# adiciona meia-janela Hann de fade-in e fade-out para suavizar o inicio e fim
janela = ss.windows.hann(1024)
ruido_pb[:512] *= janela[:512]
ruido_pb[-512:] *= janela[512:]

# %% cria o sinal AM

# f_portadora = 30        # para visualizar os graficos
f_portadora = 1000      # para auralizar o sinal atraves de falantes/fones de ouvido

# cria o sinal AM
sinal_AM, _ = gerador_seno_am(ruido_pb, fs, fc=f_portadora)

# # auralizar o sinal AM
# sd.play(0.1*sinal_AM, samplerate=fs)


fig_AM, axs_AM = plt.subplots(nrows=2, ncols=1, sharex=True)
axs_AM[0].plot(t, ruido_pb)
axs_AM[0].grid()
axs_AM[0].set_ylabel("Sinal modulador")
axs_AM[0].set_ylim([-0.5, 0.5])

axs_AM[1].plot(t, sinal_AM)
axs_AM[1].grid()
axs_AM[1].set_ylabel("Sinal modulado")
axs_AM[1].set_xlabel("Tempo [s]")

axs_AM[0].set_title("Sinal modulado em amplitude (AM)")


# %% cria o sinal FM

# sensitividade em frequencia do modulador FM (Hz/unidade do sinal modulador)
sens_freq = 50

# cria o sinal FM
sinal_FM, freq_inst, _, _ = gerador_seno_fm(ruido_pb, fs, fc=f_portadora,
                                              k=sens_freq)

# # auralizar o sinal FM
# sd.play(0.1*sinal_FM, samplerate=fs)


fig_FM, axs_FM = plt.subplots(nrows=3, ncols=1, sharex=True)
axs_FM[0].plot(t, ruido_pb)
axs_FM[0].grid()
axs_FM[0].set_ylabel("Sinal modulador")
axs_FM[0].set_ylim([-0.5, 0.5])

axs_FM[1].plot(t, sinal_FM)
axs_FM[1].grid()
axs_FM[1].set_ylabel("Sinal modulado")

axs_FM[2].plot(t, freq_inst)
axs_FM[2].grid()
axs_FM[2].set_ylabel("Freq instantanea [Hz]")
axs_FM[2].set_xlabel("Tempo [s]")
axs_FM[2].hlines(f_portadora, t[0], t[-1], colors='k', linestyles='--')
axs_FM[2].set_ylim([f_portadora - sens_freq,
                    f_portadora + sens_freq])

axs_FM[0].set_title("Sinal modulado em frequencia (FM)")

# %% criar script separado para demodular os sinais AM usando transformada de Hilbert

# analitico_AM = ss.hilbert(sinal_AM)
analitico_AM = calc_sinal_analitico(sinal_AM)

envelope_AM = np.abs(analitico_AM)

modulador_AM = envelope_AM - 1

plt.figure()
plt.subplot(211)
plt.plot(t, sinal_AM, label='Sinal AM')
plt.plot(t, envelope_AM, '--', label='Envelope')
plt.grid()
plt.legend()
plt.ylabel("Amplitude")
plt.title("Sinal AM demodulado")

plt.subplot(212)
plt.plot(t, modulador_AM, label='Sinal demodulado')
plt.plot(t, ruido_pb, '--', label='Sinal modulador original')
plt.grid()
plt.legend()
plt.ylabel("Amplitude")
plt.xlabel("Tempo [s]")

# %% criar script separado para demodular os sinais FM usando transformada de Hilbert

# analitico_FM = ss.hilbert(sinal_FM)
analitico_FM = calc_sinal_analitico(sinal_FM)

fase_instantanea_FM = np.unwrap(np.angle(analitico_FM))

freq_instantanea_FM = np.diff(fase_instantanea_FM) / (2.0*np.pi) * fs

modulador_FM = (freq_instantanea_FM - f_portadora)/sens_freq

plt.figure()

plt.plot(t[:-1], modulador_FM, label='Sinal demodulado')
plt.plot(t, ruido_pb, '--', label='Sinal modulador original')
plt.ylim([-1, 1])
plt.ylabel("Amplitude")
plt.xlabel("Tempo [s]")
plt.grid()
plt.legend()


# # %% compara a transformada de Hilbert implementada aqui com a ss.hilbert

# analitico_AM1 = ss.hilbert(sinal_AM)
# analitico_FM1 = ss.hilbert(sinal_FM)

# plt.figure()
# plt.subplot(211)
# plt.plot(analitico_AM1.real[:1000], label='Real (scipy.signal)')
# plt.plot(analitico_AM.real[:1000], '--', label='Real (calc_sinal_analitico)')
# plt.grid()
# plt.legend()
# plt.title('Sinal AM (scipy.signal vs calc_sinal_analitico)')

# plt.subplot(212)
# plt.plot(analitico_AM1.imag[:1000], label='Imag (scipy.signal)')
# plt.plot(analitico_AM.imag[:1000], '--', label='Imag (calc_sinal_analitico)')
# plt.grid()
# plt.legend()


# plt.figure()
# plt.subplot(211)
# plt.plot(analitico_FM1.real[:1000], label='Real (scipy.signal)')
# plt.plot(analitico_FM.real[:1000], '--', label='Real (calc_sinal_analitico)')
# plt.grid()
# plt.legend()
# plt.title('Sinal FM (scipy.signal vs calc_sinal_analitico)')

# plt.subplot(212)
# plt.plot(analitico_FM1.imag[:1000], label='Imag (scipy.signal)')
# plt.plot(analitico_FM.imag[:1000], '--', label='Imag (calc_sinal_analitico)')
# plt.grid()
# plt.legend()