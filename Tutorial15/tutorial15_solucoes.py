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

https://github.com/fchirono/AulasDSP

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


def am_sine_generator(xmod, fs, fc, print_m=False):
    """ Amplitude-modulated sine wave generation
    
    This function creates an amplitude-modulated (AM) signal with sinusoidal 
    carrier of frequency 'fc', and arbitrary modulating signal 'xmod'.
    The AM signal length is the same as the length of 'xmod'. 
    The carrier signal has unitary peak amplitude.

    Parameters
    ----------
    xmod: array
        Modulating signal, dim(N).
    
    fs: float
        Sampling frequency, in Hz.
    
    fc: float
        Carrier frequency, in Hz. Must be less than 'fs/2'.
    
    print_m: bool, optional
        Flag declaring whether to print the calculated modulation index.
        Default is False.
    
    Returns
    -------
    y: numpy.array
        Amplitude-modulated signal with sine carrier in Pascals, dim(N).
    m: float
        Modulation index    
        
    Warning
    -------
    spl_level must be provided in dB, ref=2e-5 Pa.
        
    Notes
    -----
    The modulation index 'm' will be equal to the peak value of the modulating
    signal 'xmod'. Its value can be printed by setting the optional flag
    'print_m' to True.
    
    For 'm' = 0.5, the carrier amplitude varies by 50% above and below its
    unmodulated level. For 'm' = 1.0, it varies by 100%. With 100% modulation 
    the wave amplitude sometimes reaches zero, and this represents full
    modulation. Increasing the modulating signal beyond that point is known as
    overmodulation.
    """
    
    assert fc < fs/2, "Carrier frequency 'fc' must be less than 'fs/2'!"
    
    Nt = xmod.shape[0]        # signal length in samples
    T = Nt/fs               # signal length in seconds
    dt = 1/fs               # sampling interval in seconds

    # vector of time samples
    t = np.linspace(0, T-dt, int(T*fs))
    
    # unit-amplitude sinusoidal carrier with frequency 'fc' [Hz]
    xc = np.sin(2*np.pi*fc*t)

    # AM signal
    y_am = (1 + xmod)*xc

    # modulation index
    m = np.max(np.abs(xmod))

    if print_m:
        print(f"AM Modulation index = {m}")
    
    if m > 1:
        print("Warning ['am_sine_generator']: modulation index m > 1\n\tSignal is overmodulated!")

    return y_am, m


def fm_sine_generator(xmod, fs, fc, k, print_info=False):
    """
    Creates a frequency-modulated (FM) signal of level 'spl_level' (in dB SPL)
    with sinusoidal carrier of frequency 'fc', arbitrary modulating signal
    'xm', frequency sensitivity 'k', and sampling frequency 'fs'. The FM signal
    length is the same as the length of 'xm'. 
    
    Parameters
    ----------
    xmod: array
        Modulating signal, dim(N)
    fs: float
        Sampling frequency, in [Hz].
    fc: float
        Carrier frequency, in [Hz]. Must be less than 'fs/2'.
    k: float
        Frequency sensitivity of the modulator. 
    print_info: bool, optional
        If True, the maximum frequency deviation and modulation index are printed. 
        Default is False
    
    Returns
    -------
    y_fm: numpy.array
        Frequency-modulated signal with sine carrier, dim(N) in [Pa].
    inst_freq: numpy.array
        Instantaneaous frequency, dim(N)
    max_freq_deviation: float
        Maximum frequency deviation [Hz]   
    FM_modulation_index: float
        Modulation index 
        
    
    Notes
    -----
    The frequency sensitivity 'k' is equal to the frequency deviation in Hz 
    away from 'fc' per unit amplitude of the modulating signal 'xmod'.
           
    """
    
    assert fc < fs/2, "Carrier frequency 'fc' must be less than 'fs/2'!"
    
     # sampling interval in seconds
    dt = 1/fs

    # instantaneous frequency of FM signal
    inst_freq = fc + k*xmod
    
    # unit-amplitude FM signal
    y_fm = np.sin(2*np.pi * np.cumsum(inst_freq)*dt)
    
    # max frequency deviation
    f_delta = k * np.max(np.abs(xmod))
    
    # FM modulation index
    m = np.max(np.abs(2*np.pi * k * np.cumsum(xmod)*dt))

    if print_info:
        print(f'\tMax freq deviation: {f_delta} Hz')
        print(f'\tFM modulation index: {m:.2f}')

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
sinal_AM, _ = am_sine_generator(ruido_pb, fs, fc=f_portadora)

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
sinal_FM, freq_inst, _, _ = fm_sine_generator(ruido_pb, fs, fc=f_portadora,
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
plt.subplot(211)
plt.plot(t[:-1], freq_instantanea_FM, label='Freq inst demodulada')
plt.plot(t, freq_inst, '--', label='Freq inst original')
plt.ylim([f_portadora - sens_freq,
          f_portadora + sens_freq])
plt.grid()
plt.legend()
plt.ylabel('Freq [Hz]')
plt.title("Sinal FM demodulado")

plt.subplot(212)
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