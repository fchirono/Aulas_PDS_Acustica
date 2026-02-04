"""
Exemplo de solução para o Tutorial 13 de Processamento Digital de Sinais

https://github.com/fchirono/AulasDSP

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile

import matplotlib.pyplot as plt
plt.close('all')


# flag para escrita de arquivos wav
escrever_wav = False

# flag para salvar figuras
salvar_fig = False

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#  Parte 1 - Removendo uma perturbação de um sinal de áudio
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# ler um arquivo wave e propriedades adicionais, i.e. sua frequência de amostragem
fs, sinal_16bits = wavfile.read("BetterDaysAheadT.wav")

# confirmar o tipo de dados obtidos do arquivo wav
sinal_16bits.dtype

# Considerar arquivo de 16 bits
N_bits = 16

# normalização dos dados de áudio de 16 bits
sinal_esquerdo = sinal_16bits[:, 0]/(2.**(N_bits - 1)-1)
sinal_direito = sinal_16bits[:, 1]/(2.**(N_bits - 1)-1)

dt = 1./fs                          # resolução temporal
N_dft = 2**15                       # tamanho da DFT
df = float(fs)/N_dft                # resolução de frequência
f = np.linspace(0, fs-df, N_dft)    # vetor de frequência

# Calcular a FFT de "N_dft" pontos
sinal_esquerdo_f = np.fft.fft(sinal_esquerdo, N_dft)
sinal_direito_f = np.fft.fft(sinal_direito, N_dft)


# Plotar o meio-espectro para os sinais
plt.figure()
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         label='Canal Esquerdo')
plt.xlim(0, fs/2)
plt.legend()
plt.title('BetterDaysAheadT.wav (canal esquerdo)')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

# plotar uma linha vertical em 4 kHz
plt.vlines(4000, -80, 60, colors='k', linestyle='dashdot')
plt.text(4300, -50, '4 kHz')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calcular filtros FIR para remover a perturbação
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# número de coeficientes do filtro - usar número ímpar de coeficientes para filtro passa-altas
N_FIR = 201

# frequência de corte para passa-baixas
f0_pb = 3500

# frequência de corte normalizada (para METADE da frequência de amostragem)
w0_pb = f0_pb/(fs/2.)

# calcular um filtro passa-baixas
b_passabaixas = signal.firwin(N_FIR, w0_pb)

# frequência de corte para passa-altas
f0_pa = 4500
w0_pa = f0_pa/(fs/2.)

# calcular um filtro passa-altas (um filtro passa-altas irá rejeitar a "frequência
# zero", e portanto essa frequência não deve "passar")
b_passaaltas = signal.firwin(N_FIR, w0_pa, pass_zero=False)


# calcular a resposta ao impulso FIR
a_FIR = np.zeros(b_passabaixas.shape)
a_FIR[0] = 1.
FIR_pb_tupla = (b_passabaixas, a_FIR, dt)
t_RI, RI_FIR_pb_tupla = signal.dimpulse(FIR_pb_tupla, n=N_dft)
RI_FIR_pb = RI_FIR_pb_tupla[0]

FIR_pa_tupla = (b_passaaltas, a_FIR, dt)
t_RI, RI_FIR_pa_tupla = signal.dimpulse(FIR_pa_tupla, n=N_dft)
RI_FIR_pa = RI_FIR_pa_tupla[0]


# calcular a resposta em frequência 'h' dos filtros em suas respectivas
# frequências normalizadas 'w' (0 < w < pi) com 'N_dft//2' pontos
w_pb, H_FIR_pb = signal.freqz(b_passabaixas, a_FIR, N_dft//2)
w_pa, H_FIR_pa = signal.freqz(b_passaaltas, a_FIR, N_dft//2)


# calcular zeros dos filtros FIR
zeros_fir_pb = np.roots(b_passabaixas)
zeros_fir_pa = np.roots(b_passaaltas)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calcular um filtro IIR passa-baixas para remover a perturbação

# Parâmetros de Desempenho do Filtro
# frequência de borda da faixa de passagem
f_pass_pb = 3000
w_pass_pb = f_pass_pb/(fs/2.)    # normalizada para fs/2

# perda máxima na faixa de passagem [em dB]
g_pass_pb = 3.

# frequência de borda da faixa de rejeição
f_stop_pb = 4000
w_stop_pb = f_stop_pb/(fs/2.)    # normalizada para fs/2

# atenuação mínima na faixa de rejeição [em dB]
g_stop_pb = 40.

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cálculo de filtros Butterworth - PASSA-BAIXAS

# calcular ordem do filtro para os requisitos acima
N_butter_pb, w0_butter_pb = signal.buttord(w_pass_pb, w_stop_pb,
                                           g_pass_pb, g_stop_pb)

# calcular filtros passa-baixas
b_butter_pb, a_butter_pb = signal.butter(N_butter_pb, w0_butter_pb,
                                         btype='lowpass')

# obter resposta ao impulso
butter_pb_tupla = (b_butter_pb, a_butter_pb, dt)
t_RI, RI_butter_pb_tupla = signal.dimpulse(butter_pb_tupla, n=N_dft)
RI_butter_pb = RI_butter_pb_tupla[0]

# calcular polos e zeros
zeros_butter_pb = np.roots(b_butter_pb)
polos_butter_pb = np.roots(a_butter_pb)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cálculo de filtros Chebyshev Tipo 1 - PASSA-BAIXAS

# oscilação máxima na faixa de passagem [dB]
r1 = 3.

# calcular ordem do filtro para os requisitos acima
N_cheby1_pb, w0_cheby1_pb = signal.cheb1ord(w_pass_pb, w_stop_pb, g_pass_pb,
                                            g_stop_pb)

# calcular filtros passa-baixas
b_cheby1_pb, a_cheby1_pb = signal.cheby1(N_cheby1_pb, r1, w0_cheby1_pb,
                                         btype='lowpass')

# obter resposta ao impulso do filtro
cheby1_pb_tupla = (b_cheby1_pb, a_cheby1_pb, dt)
t_RI, RI_cheby1_pb_tupla = signal.dimpulse(cheby1_pb_tupla, n=N_dft)
RI_cheby1_pb = RI_cheby1_pb_tupla[0]

# calcular os polos e zeros
zeros_cheby1_pb = np.roots(b_cheby1_pb)
polos_cheby1_pb = np.roots(a_cheby1_pb)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cálculo de filtros Chebyshev Tipo 2 - PASSA-BAIXAS

# atenuação mínima na faixa de rejeição [dB]
r2 = 40.

# calcular ordem do filtro para os requisitos acima
N_cheby2_pb, w0_cheby2_pb = signal.cheb2ord(w_pass_pb, w_stop_pb, g_pass_pb,
                                            g_stop_pb)

# calcular filtros passa-baixas
b_cheby2_pb, a_cheby2_pb = signal.cheby2(N_cheby2_pb, r2, w0_cheby2_pb,
                                         btype='lowpass')

# obter resposta ao impulso do filtro
cheby2_pb_tupla = (b_cheby2_pb, a_cheby2_pb, dt)
t_RI, RI_cheby2_pb_tupla = signal.dimpulse(cheby2_pb_tupla, n=N_dft)
RI_cheby2_pb = RI_cheby2_pb_tupla[0]

# calcular os polos e zeros
zeros_cheby2_pb = np.roots(b_cheby2_pb)
polos_cheby2_pb = np.roots(a_cheby2_pb)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# calcular a resposta em frequência 'h' dos filtros em suas respectivas
# frequências normalizadas 'w' (0 < w < pi) com 'N_dft//2' pontos
w, H_butter_pb = signal.freqz(b_butter_pb, a_butter_pb, N_dft//2)
w, H_cheby1_pb = signal.freqz(b_cheby1_pb, a_cheby1_pb, N_dft//2)
w, H_cheby2_pb = signal.freqz(b_cheby2_pb, a_cheby2_pb, N_dft//2)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plotar o diagrama polo-zero dos filtros
fig_pz = plt.figure(figsize=(12, 12))

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# FIR
ax_fir_pb_pz = fig_pz.add_subplot(221)
plt.axis('equal')
plt.plot(np.real(zeros_fir_pb), np.imag(zeros_fir_pb), 'ob',
         label='Zeros')
plt.legend()

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_fir_pb_pz.add_artist(circulo_unitario)

# ax_fir_pb_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_fir_pb_pz.set_xlim([-1.5, 1.5])


ax_fir_pb_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')
ax_fir_pb_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
plt.title('Filtro FIR Passa-Baixas, ordem {:d}'.format(N_FIR))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Butter
ax_butter_pb_pz = fig_pz.add_subplot(222)
plt.plot(np.real(zeros_butter_pb), np.imag(zeros_butter_pb), 'ob',
         label='Zeros')
plt.plot(np.real(polos_butter_pb), np.imag(polos_butter_pb), 'xr',
         label='Polos')

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_butter_pb_pz.add_artist(circulo_unitario)

ax_butter_pb_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
ax_butter_pb_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')

plt.axis('equal')
# ax_butter_pb_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_butter_pb_pz.set_xlim([-1.5, 1.5])

plt.title('Filtro Butterworth Passa-Baixas, ordem {:d}'.format(N_butter_pb))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend(loc='upper right')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cheby1
ax_cheby1_pb_pz = fig_pz.add_subplot(223)
plt.plot(np.real(zeros_cheby1_pb), np.imag(zeros_cheby1_pb), 'ob',
         label='Zeros')
plt.plot(np.real(polos_cheby1_pb), np.imag(polos_cheby1_pb), 'xr',
         label='Polos')

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_cheby1_pb_pz.add_artist(circulo_unitario)

ax_cheby1_pb_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
ax_cheby1_pb_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')

plt.axis('equal')
# ax_cheby1_pb_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_cheby1_pb_pz.set_xlim([-1.5, 1.5])

plt.title('Filtro Chebyshev Tipo 1 Passa-Baixas, ordem {:d}'.format(N_cheby1_pb))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend(loc='upper right')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cheby2
ax_cheby2_pb_pz = fig_pz.add_subplot(224)
plt.plot(np.real(zeros_cheby2_pb), np.imag(zeros_cheby2_pb), 'ob',
         label='Zeros')
plt.plot(np.real(polos_cheby2_pb), np.imag(polos_cheby2_pb), 'xr',
         label='Polos')

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_cheby2_pb_pz.add_artist(circulo_unitario)

ax_cheby2_pb_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
ax_cheby2_pb_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')

plt.axis('equal')
# ax_cheby2_pb_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_cheby2_pb_pz.set_xlim([-1.5, 1.5])

plt.title('Filtro Chebyshev Tipo 2 Passa-Baixas, ordem {:d}'.format(N_cheby2_pb))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend(loc='upper right')

plt.tight_layout()

if salvar_fig:
    plt.savefig('IIR_pb_pz.png')

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plotar a resposta em frequência dos filtros passa-baixas
plt.figure(figsize=(8, 10))
plt.subplot(311)
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_FIR_pb)), 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_butter_pb)), 'C0',
         label='Butter, ordem {:d}'.format(N_butter_pb))
plt.title('Resposta em Frequência Filtros FIR vs. IIR Passa-Baixas')
plt.ylabel('Magnitude [dB]')
plt.xlim([0, fs/2])
plt.ylim([-80, 10])
plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_FIR_pb)), 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_cheby1_pb)), 'C0',
         label='Cheby1, ordem {:d}'.format(N_cheby1_pb))
plt.ylabel('Magnitude [dB]')
plt.xlim([0, fs/2])
plt.ylim([-80, 10])
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_FIR_pb)), 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_cheby2_pb)), 'C0',
         label='Cheby2, ordem {:d}'.format(N_cheby2_pb))
plt.ylabel('Magnitude [dB]')
plt.xlim([0, fs/2])
plt.ylim([-80, 10])
plt.legend(loc='upper right')

plt.xlabel('Frequência [Hz]')
plt.tight_layout()

if salvar_fig:
    plt.savefig('IIR_pb_RespFreq.png')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plotar a resposta ao impulso dos filtros passa-baixas
plt.figure(figsize=(8, 10))
plt.subplot(311)
plt.plot(t_RI[:250], RI_FIR_pb[:250], 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(t_RI[:250], RI_butter_pb[:250], 'C0',
         label='Butter, ordem {:d}'.format(N_butter_pb))
plt.title('Resposta ao Impulso Filtros FIR vs. IIR Passa-Baixas')
plt.ylabel('Amplitude')
plt.xlim([0, t_RI[250]])
plt.ylim([-0.1, 0.2])
plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(t_RI[:250], RI_FIR_pb[:250], 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(t_RI[:250], RI_cheby1_pb[:250], 'C0',
         label='Cheby1, ordem {:d}'.format(N_cheby1_pb))
plt.ylabel('Amplitude')
plt.xlim([0, t_RI[250]])
plt.ylim([-0.1, 0.2])
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(t_RI[:250], RI_FIR_pb[:250], 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(t_RI[:250], RI_cheby2_pb[:250], 'C0',
         label='Cheby2, ordem {:d}'.format(N_cheby1_pb))
plt.ylabel('Amplitude')
plt.xlim([0, t_RI[250]])
plt.ylim([-0.1, 0.2])
plt.legend(loc='upper right')

plt.xlabel('Tempo [s]')
plt.tight_layout()

if salvar_fig:
    plt.savefig('IIR_pb_RespImp.png')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calcular um filtro IIR passa-altas para remover a perturbação

# Parâmetros de Desempenho do Filtro
# frequência de borda da faixa de passagem
f_pass_pa = 5000
w_pass_pa = f_pass_pa/(fs/2.)    # normalizada para fs/2

# perda máxima na faixa de passagem [em dB]
g_pass_pa = 3.

# frequência de borda da faixa de rejeição
f_stop_pa = 4000
w_stop_pa = f_stop_pa/(fs/2.)    # normalizada para fs/2

# atenuação mínima na faixa de rejeição [em dB]
g_stop_pa = 40.

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cálculo de filtros Butterworth - PASSA-ALTAS

# calcular ordem do filtro para os requisitos acima
N_butter_pa, w0_butter_pa = signal.buttord(w_pass_pa, w_stop_pa,
                                           g_pass_pa, g_stop_pa)

# calcular filtros passa-altas
b_butter_pa, a_butter_pa = signal.butter(N_butter_pa, w0_butter_pa,
                                         btype='highpass')

# obter resposta ao impulso do filtro
butter_pa_tupla = (b_butter_pa, a_butter_pa, dt)
t_RI, RI_butter_pa_tupla = signal.dimpulse(butter_pa_tupla, n=N_dft)
RI_butter_pa = RI_butter_pa_tupla[0]

# calcular os polos e zeros
zeros_butter_pa = np.roots(b_butter_pa)
polos_butter_pa = np.roots(a_butter_pa)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cálculo de filtros Chebyshev Tipo 1 - PASSA-ALTAS

# oscilação máxima na faixa de passagem [dB]
r1 = 6.

# calcular ordem do filtro para os requisitos acima
N_cheby1_pa, w0_cheby1_pa = signal.cheb1ord(w_pass_pa, w_stop_pa, g_pass_pa,
                                            g_stop_pa)

# calcular filtros passa-altas
b_cheby1_pa, a_cheby1_pa = signal.cheby1(N_cheby1_pa, r1, w0_cheby1_pa,
                                         btype='highpass')

# obter resposta ao impulso do filtro
cheby1_pa_tupla = (b_cheby1_pa, a_cheby1_pa, dt)
t_RI, RI_cheby1_pa_tupla = signal.dimpulse(cheby1_pa_tupla, n=N_dft)
RI_cheby1_pa = RI_cheby1_pa_tupla[0]

# calcular os polos e zeros
zeros_cheby1_pa = np.roots(b_cheby1_pa)
polos_cheby1_pa = np.roots(a_cheby1_pa)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cálculo de filtros Chebyshev Tipo 2 - PASSA-ALTAS

# atenuação mínima na faixa de rejeição [dB]
r2 = 40.

# calcular ordem do filtro para os requisitos acima
N_cheby2_pa, w0_cheby2_pa = signal.cheb2ord(w_pass_pa, w_stop_pa, g_pass_pa,
                                            g_stop_pa)

# calcular filtros passa-altas
b_cheby2_pa, a_cheby2_pa = signal.cheby2(N_cheby2_pa, r2, w0_cheby2_pa,
                                         btype='highpass')

# obter resposta ao impulso do filtro
cheby2_pa_tupla = (b_cheby2_pa, a_cheby2_pa, dt)
t_RI, RI_cheby2_pa_tupla = signal.dimpulse(cheby2_pa_tupla, n=N_dft)
RI_cheby2_pa = RI_cheby2_pa_tupla[0]

# calcular os polos e zeros
zeros_cheby2_pa = np.roots(b_cheby2_pa)
polos_cheby2_pa = np.roots(a_cheby2_pa)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# calcular a resposta em frequência 'h' dos filtros em suas respectivas
# frequências normalizadas 'w' (0 < w < pi) com 'N_dft//2' pontos
w, H_butter_pa = signal.freqz(b_butter_pa, a_butter_pa, N_dft//2)
w, H_cheby1_pa = signal.freqz(b_cheby1_pa, a_cheby1_pa, N_dft//2)
w, H_cheby2_pa = signal.freqz(b_cheby2_pa, a_cheby2_pa, N_dft//2)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plotar o diagrama polo-zero dos filtros
fig_pz2 = plt.figure(figsize=(12, 12))

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# FIR
ax_fir_pa_pz = fig_pz2.add_subplot(221)
plt.axis('equal')
plt.plot(np.real(zeros_fir_pa), np.imag(zeros_fir_pa), 'ob',
         label='Zeros')
plt.legend()

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_fir_pa_pz.add_artist(circulo_unitario)
# ax_fir_pa_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_fir_pa_pz.set_xlim([-1.5, 1.5])

ax_fir_pa_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
ax_fir_pa_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')
plt.title('Filtro FIR Passa-Altas, ordem {:d}'.format(N_FIR))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Butter
ax_butter_pa_pz = fig_pz2.add_subplot(222)
plt.plot(np.real(zeros_butter_pa), np.imag(zeros_butter_pa), 'ob',
         label='Zeros')
plt.plot(np.real(polos_butter_pa), np.imag(polos_butter_pa), 'xr',
         label='Polos')

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_butter_pa_pz.add_artist(circulo_unitario)

ax_butter_pa_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
ax_butter_pa_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')

plt.axis('equal')
# ax_butter_pa_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_butter_pa_pz.set_xlim([-1.5, 1.5])

plt.title('Filtro Butterworth Passa-Altas, ordem {:d}'.format(N_butter_pa))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend(loc='upper right')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cheby1
ax_cheby1_pa_pz = fig_pz2.add_subplot(223)
plt.plot(np.real(zeros_cheby1_pa), np.imag(zeros_cheby1_pa), 'ob',
         label='Zeros')
plt.plot(np.real(polos_cheby1_pa), np.imag(polos_cheby1_pa), 'xr',
         label='Polos')

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_cheby1_pa_pz.add_artist(circulo_unitario)

ax_cheby1_pa_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
ax_cheby1_pa_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')

plt.axis('equal')
# ax_cheby1_pa_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_cheby1_pa_pz.set_xlim([-1.5, 1.5])

plt.title('Filtro Chebyshev Tipo 1 Passa-Altas, ordem {:d}'.format(N_cheby1_pa))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend(loc='upper right')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Cheby2
ax_cheby2_pa_pz = fig_pz2.add_subplot(224)
plt.plot(np.real(zeros_cheby2_pa), np.imag(zeros_cheby2_pa), 'ob',
         label='Zeros')
plt.plot(np.real(polos_cheby2_pa), np.imag(polos_cheby2_pa), 'xr',
         label='Polos')

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_cheby2_pa_pz.add_artist(circulo_unitario)

ax_cheby2_pa_pz.hlines(0, -1.5, 1.5, linestyle='dashdot')
ax_cheby2_pa_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')

plt.axis('equal')
# ax_cheby2_pa_pz.axis([-1.5, 1.5, -1.5, 1.5])
ax_cheby2_pa_pz.set_xlim([-1.5, 1.5])

plt.title('Filtro Chebyshev Tipo 2 Passa-Altas, ordem {:d}'.format(N_cheby2_pa))
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend(loc='upper right')

plt.tight_layout()

if salvar_fig:
    plt.savefig('IIR_pa_pz.png')

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plotar a resposta em frequência dos filtros passa-altas
plt.figure(figsize=(8, 10))
plt.subplot(311)
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_FIR_pa)), 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_butter_pa)), 'C1',
         label='Butter, ordem {:d}'.format(N_butter_pa))
plt.title('Resposta em Frequência Filtros FIR vs. IIR Passa-Altas')
plt.ylabel('Magnitude [dB]')
plt.xlim([0, fs/2])
plt.ylim([-80, 10])
plt.legend(loc='lower right')

plt.subplot(312)
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_FIR_pa)), 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_cheby1_pa)), 'C1',
         label='Cheby1, ordem {:d}'.format(N_cheby1_pa))
plt.ylabel('Magnitude [dB]')
plt.xlim([0, fs/2])
plt.ylim([-80, 10])
plt.legend(loc='lower right')

plt.subplot(313)
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_FIR_pa)), 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H_cheby2_pa)), 'C1',
         label='Cheby2, ordem {:d}'.format(N_cheby2_pa))
plt.ylabel('Magnitude [dB]')
plt.xlim([0, fs/2])
plt.ylim([-80, 10])
plt.legend(loc='lower right')

plt.xlabel('Frequência [Hz]')
plt.tight_layout()

if salvar_fig:
    plt.savefig('IIR_pa_RespFreq.png')

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plotar a resposta ao impulso dos filtros passa-altas
plt.figure(figsize=(8, 10))
plt.subplot(311)
plt.plot(t_RI[:250], RI_FIR_pa[:250], 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(t_RI[:250], RI_butter_pa[:250], 'C1',
         label='Butter, ordem {:d}'.format(N_butter_pa))
plt.title('Resposta ao Impulso Filtros FIR vs. IIR Passa-Altas')
plt.ylabel('Amplitude')
plt.xlim([0, t_RI[250]])
plt.ylim([-0.8, 1])
plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(t_RI[:250], RI_FIR_pa[:250], 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(t_RI[:250], RI_cheby1_pa[:250], 'C1',
         label='Cheby1, ordem {:d}'.format(N_cheby1_pa))
plt.ylabel('Amplitude')
plt.xlim([0, t_RI[250]])
plt.ylim([-0.8, 1])
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(t_RI[:250], RI_FIR_pa[:250], 'k-.',
         label='FIR, ordem {:d}'.format(N_FIR))
plt.plot(t_RI[:250], RI_cheby2_pa[:250], 'C1',
         label='Cheby2, ordem {:d}'.format(N_cheby1_pa))
plt.ylabel('Amplitude')
plt.xlim([0, t_RI[250]])
plt.ylim([-0.8, 1])
plt.legend(loc='upper right')

plt.xlabel('Tempo [s]')
plt.tight_layout()

if salvar_fig:
    plt.savefig('IIR_pa_RespImp.png')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# aplicar os filtros
sinal_butter_pb = signal.lfilter(b_butter_pb, a_butter_pb, sinal_esquerdo)
sinal_cheby1_pb = signal.lfilter(b_cheby1_pb, a_cheby1_pb, sinal_esquerdo)
sinal_cheby2_pb = signal.lfilter(b_cheby2_pb, a_cheby2_pb, sinal_esquerdo)

sinal_butter_pa = signal.lfilter(b_butter_pa, a_butter_pa, sinal_esquerdo)
sinal_cheby1_pa = signal.lfilter(b_cheby1_pa, a_cheby1_pa, sinal_esquerdo)
sinal_cheby2_pa = signal.lfilter(b_cheby2_pa, a_cheby2_pa, sinal_esquerdo)

# calcular a DFT
sinal_butter_pb_f = np.fft.fft(sinal_butter_pb, N_dft)
sinal_cheby1_pb_f = np.fft.fft(sinal_cheby1_pb, N_dft)
sinal_cheby2_pb_f = np.fft.fft(sinal_cheby2_pb, N_dft)

sinal_butter_pa_f = np.fft.fft(sinal_butter_pa, N_dft)
sinal_cheby1_pa_f = np.fft.fft(sinal_cheby1_pa, N_dft)
sinal_cheby2_pa_f = np.fft.fft(sinal_cheby2_pa, N_dft)


# plotar os espectros filtrados
plt.figure()
plt.subplot(211)
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_butter_pb_f[:N_dft//2+1])),
         label='Butter PB')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado Butterworth passa-baixas')
plt.ylabel('Magnitude [dB]')

plt.subplot(212)
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_butter_pa_f[:N_dft//2+1])),
         'C1', label='Butter PA')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado Butterworth passa-altas')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')
plt.tight_layout()

if salvar_fig:
    plt.savefig('PB_PA_Butter.png')


plt.figure()
plt.subplot(211)
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_cheby1_pb_f[:N_dft//2+1])),
         label='Cheby1 PB')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado Chebyshev Tipo 1 passa-baixas')
plt.ylabel('Magnitude [dB]')

plt.subplot(212)
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_cheby1_pa_f[:N_dft//2+1])),
         'C1', label='Cheby1 PA')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado Chebyshev Tipo 1 passa-altas')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')
plt.tight_layout()

if salvar_fig:
    plt.savefig('PB_PA_Cheby1.png')


plt.figure()
plt.subplot(211)
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_cheby2_pb_f[:N_dft//2+1])),
         label='Cheby2 PB')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado Chebyshev Tipo 2 passa-baixas')
plt.ylabel('Magnitude [dB]')

plt.subplot(212)
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_cheby2_pa_f[:N_dft//2+1])),
         'C1', label='Cheby2 PA')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado Chebyshev Tipo 2 passa-altas')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')
plt.tight_layout()

if salvar_fig:
    plt.savefig('PB_PA_Cheby2.png')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#  Somar os sinais filtrados passa-baixas e passa-altas
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

sinal_butter_soma = sinal_butter_pb + sinal_butter_pa
sinal_cheby1_soma = sinal_cheby1_pb + sinal_cheby1_pa
sinal_cheby2_soma = sinal_cheby2_pb + sinal_cheby2_pa

sinal_butter_soma_f = np.fft.fft(sinal_butter_soma, N_dft)
sinal_cheby1_soma_f = np.fft.fft(sinal_cheby1_soma, N_dft)
sinal_cheby2_soma_f = np.fft.fft(sinal_cheby2_soma, N_dft)

plt.figure()
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_butter_soma_f[:N_dft//2+1])),
         label='Butter PB+PA')
plt.xlim(0, fs/2)
plt.legend()
plt.title('Espectros pré e pós-filtragem usando Filtros Butterworth')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

if salvar_fig:
    plt.savefig('PB_mais_PA_Butter.png')


plt.figure()
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_cheby1_soma_f[:N_dft//2+1])),
         label='Cheby1 PB+PA')
plt.xlim(0, fs/2)
plt.legend()
plt.title('Espectros pré e pós-filtragem usando Filtros Chebyshev Tipo 1')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

if salvar_fig:
    plt.savefig('PB_mais_PA_Cheby1.png')


plt.figure()
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1], 20*np.log10(np.abs(sinal_cheby2_soma_f[:N_dft//2+1])),
         label='Cheby2 PB+PA')
plt.xlim(0, fs/2)
plt.legend()
plt.title('Espectros pré e pós-filtragem usando Filtros Chebyshev Tipo 2')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

if salvar_fig:
    plt.savefig('PB_mais_PA_Cheby2.png')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#  Parte 2 - Projeto de Filtro Baseado em Diagrama de Blocos
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# completar o numerador para ter o mesmo comprimento do denominador
b_bloco = np.array([1, 2, 0])
a_bloco = np.array([1, -1, 0.5])

# tupla definindo o diagrama de blocos
dt_bloco = 1
bloco_tupla = (b_bloco, a_bloco, dt_bloco)

# calculando a resposta ao impulso
n_bloco, RI_bloco_tupla = signal.dimpulse(bloco_tupla, n=30)
RI_bloco = RI_bloco_tupla[0]

plt.figure()
plt.plot([0, 1, 2], [1, 3, 2.5], 'ro', markersize=12, label='Teórico')
plt.stem(n_bloco, RI_bloco, label='signal.dimpulse')
plt.title('Resposta ao Impulso do Diagrama de Blocos')
plt.xlabel('n [amostras]')
plt.ylabel('Amplitude')
plt.axis([-1, 30, -1, 3.5])
plt.legend()

if salvar_fig:
    plt.savefig('DiagramaBlocosRI.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# calculando a resposta em frequência
w, H1 = signal.freqz(b_bloco, a_bloco, N_dft)

H2 = (1 + 2*np.exp(-1j*w))/(1 - np.exp(-1j*w) + 0.5*np.exp(-2j*w))

plt.figure()
plt.subplot(211)
plt.plot(w/np.pi, 20*np.log10(np.abs(H2)), 
         linewidth=3, label='Teórico')
plt.plot(w/np.pi, 20*np.log10(np.abs(H1)),
         linestyle='--', label='signal.freqz')
plt.ylabel('Magnitude [dB]')
plt.title('Resposta em Frequência do Diagrama de Blocos')
plt.grid()
plt.legend()

plt.subplot(212)
plt.plot(w/np.pi, np.angle(H2), 
         linewidth=3, label='Teórico')
plt.plot(w/np.pi, np.angle(H1), 
         linestyle='--', label='signal.freqz')
plt.xlabel('Frequência Normalizada [' + r'$\times \pi$' + ' rad/amostra]')
plt.ylabel('Fase [rad]')
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.legend()

if salvar_fig:
    plt.savefig('DiagramaBlocosRespFreq.png')


plt.figure()
plt.subplot(211)
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H2)), 
         linewidth=3, label='Teórico')
plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(H1)), 
         linestyle='--', label='signal.freqz')
plt.ylabel('Magnitude [dB]')
plt.xlim([0, fs/2])
plt.grid()
plt.legend()
plt.title('Resposta em Frequência do Diagrama de Blocos (fs = 44100 Hz)')


plt.subplot(212)
plt.plot(w*fs/(2*np.pi), np.angle(H2), 
         linewidth=3, label='Teórico')
plt.plot(w*fs/(2*np.pi), np.angle(H1), 
         linestyle='--', label='signal.freqz')
plt.xlabel('Frequência [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim([0, fs/2])
plt.ylim([-np.pi, np.pi])
plt.grid()

plt.legend()

if salvar_fig:
    plt.savefig('DiagramaBlocosRespFreq_Hz.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# calculando o diagrama polo-zero

zeros_bloco = np.roots(b_bloco)
polos_bloco = np.roots(a_bloco)

fig_bloco_pz = plt.figure()
ax_bloco_pz = fig_bloco_pz.add_subplot(111)
plt.axis('equal')
ax_bloco_pz.plot(np.real(zeros_bloco), np.imag(zeros_bloco), 'bo',
                 label='Zeros')
ax_bloco_pz.plot(np.real(polos_bloco), np.imag(polos_bloco), 'rx',
                 label='Polos')
plt.legend()

# plotar círculo unitário e eixos (Re, Im)
circulo_unitario = plt.Circle((0, 0), 1, fc='none', ec='k', linestyle='dashdot')
ax_bloco_pz.add_artist(circulo_unitario)
ax_bloco_pz.axis([-1.5, 1.5, -1.5, 1.5])

ax_bloco_pz.hlines(0, -2.5, 1.5, linestyle='dashdot')
ax_bloco_pz.vlines(0, -1.5, 1.5, linestyle='dashdot')

plt.axis('equal')
ax_bloco_pz.axis([-2.5, 1.5, -1.5, 1.5])

plt.title('Polos e Zeros do Diagrama de Blocos')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.legend(loc='upper left')

if salvar_fig:
    plt.savefig('DiagramaBlocosPZ.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Filtrar o sinal de música

# escrever_wav=True

# ler um arquivo wavefile
fs, musica_16bits = wavfile.read("BetterDaysAhead.wav")

# Considerar arquivo de 16 bits
N_bits = 16

# normalização dos dados de áudio de 16 bits
musica = musica_16bits[:, 0]/(2.**(N_bits - 1)-1)

# filtrando o sinal de música
musica_filtrada = signal.lfilter(b_bloco, a_bloco, musica)

# verificar flag de escrita
if escrever_wav is True:
    # Amplitude de pico da amostra wav
    pico_wav = 2**(N_bits-1) - 1

    # normalizando os sinais para amplitude unitária
    musica_filt_norm = musica_filtrada/np.max(np.abs(musica_filtrada))

    # Converter para int de 16 bits e escrever os arquivos wav dos sinais gerados
    wavfile.write('musica_filtrada.wav', fs, np.int16(musica_filt_norm*pico_wav))
    