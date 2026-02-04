"""
Exemplo de solução para o Tutorial 12 de Processamento Digital de Sinais

https://github.com/fchirono/AulasDSP

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
import scipy.signal as ss
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

# ler um arquivo wav e frequência de amostragem
fs, sinal_16bits = wavfile.read("BetterDaysAheadT.wav")

# confirmar o tipo de dados obtidos do arquivo wav
sinal_16bits.dtype

# Considerar arquivo de 16 bits
N_bits = 16

# normalização dos dados de áudio de 16 bits
sinal_esquerdo = sinal_16bits[:, 0]/(2.**(N_bits - 1)-1)
sinal_direito = sinal_16bits[:, 1]/(2.**(N_bits - 1)-1)

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
plt.title('BetterDaysAheadT.wav')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

# plotar uma linha vertical em 4 kHz
plt.vlines(4000, -80, 60, colors='k', linestyle='dotted')
plt.text(4300, -50, '4 kHz')

if salvar_fig:
    plt.savefig('BetterDaysAheadT_espectro.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calcular um filtro FIR passa-baixas para remover a perturbação
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# número de coeficientes do filtro - usar número ímpar de coeficientes para filtro passa-altas
N_coefs = 201

# frequência de corte para passa-baixas
f0_pb = 3500

# frequência de corte normalizada (para METADE da frequência de amostragem)
w0_pb = f0_pb/(fs/2.)

# calcular um filtro passa-baixas
b_passabaixas = ss.firwin(N_coefs, w0_pb)

a_passabaixas = np.zeros(b_passabaixas.shape)
a_passabaixas[0] = 1.

# frequência de corte para passa-altas
f0_pa = 4500
w0_pa = f0_pa/(fs/2.)

# calcular um filtro passa-altas ("pass_zero=False" é usado para designar filtro passa-alta)
b_passaaltas = ss.firwin(N_coefs, w0_pa, pass_zero=False)

a_passaaltas = np.zeros(b_passaaltas.shape)
a_passaaltas[0] = 1.

# calcular a resposta em frequência 'h' dos filtros em suas respectivas
# frequências normalizadas 'w' (0 < w < pi) com 'N_dft//2 + 1' pontos
w_pb, H_pb = ss.freqz(b_passabaixas, a_passabaixas, N_dft//2 + 1)
w_pa, H_pa = ss.freqz(b_passaaltas, a_passaaltas, N_dft//2 + 1)

plt.figure()
plt.plot(w_pb/np.pi, 20*np.log10(np.abs(H_pb)), label='Passa-Baixas')
plt.plot(w_pa/np.pi, 20*np.log10(np.abs(H_pa)), '--', label='Passa-Altas')
plt.title('Resposta em frequência dos filtros')
plt.xlabel('Frequência normalizada [' + r'$\times \pi$' + ' rad/amostra]')
plt.ylabel('Magnitude [dB]')
plt.ylim([-90, 10])
plt.legend(loc='center right')

if salvar_fig:
    plt.savefig('Filtros_RespFreq.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# filtrar o canal esquerdo
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# aplicar os filtros
sinal_esquerdo_pb = ss.lfilter(b_passabaixas, a_passabaixas, sinal_esquerdo)
sinal_esquerdo_pa = ss.lfilter(b_passaaltas, a_passaaltas, sinal_esquerdo)

# calcular a DFT
sinal_esquerdo_pb_f = np.fft.fft(sinal_esquerdo_pb, N_dft)
sinal_esquerdo_pa_f = np.fft.fft(sinal_esquerdo_pa, N_dft)

# plotar os espectros filtrados
plt.figure()
plt.subplot(211)
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_pb_f[:N_dft//2+1])),
         label='Filtrado PB')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado passa-baixas')
plt.ylabel('Magnitude [dB]')

plt.subplot(212)
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Original')
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_pa_f[:N_dft//2+1])),
         color='C1', label='Filtrado PA')
plt.xlim(0, fs/2)
plt.ylim([-80, 60])
plt.legend()
plt.title('Sinal filtrado passa-altas')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

plt.tight_layout()

if salvar_fig:
    plt.savefig('Sinais_filtrados.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#  dividir o espectro de saída pelo espectro de entrada
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

razao_espectro_pb = sinal_esquerdo_pb_f/sinal_esquerdo_f
razao_espectro_pa = sinal_esquerdo_pa_f/sinal_esquerdo_f

# plotar a razão entre os espectros de saída e entrada
plt.figure()
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(razao_espectro_pb[:N_dft//2+1])),
         label='Passa-baixas')
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(razao_espectro_pa[:N_dft//2+1])),
         'C1', linestyle='--', label='Passa-altas')
plt.xlim(0, fs/2)
plt.ylim(-100, 20)
plt.legend(loc='lower right')
plt.title('Razão dos espectros do sinal')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

if salvar_fig:
    plt.savefig('razao_espectros_sinal.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#  Somar os sinais filtrados passa-baixas e passa-altas
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

sinal_esquerdo_soma = sinal_esquerdo_pb + sinal_esquerdo_pa
sinal_esquerdo_soma_f = np.fft.fft(sinal_esquerdo_soma, N_dft)

plt.figure()
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_f[:N_dft//2+1])),
         color='lightgray', linestyle=':', label='Canal Esquerdo Original')
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_soma_f[:N_dft//2+1])),
         'C0', label='Passa-baixas + Passa-altas')
plt.xlim(0, fs/2)
plt.legend()
plt.title('Espectros pré e pós-filtragem')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

if salvar_fig:
    plt.savefig('PB_PA_soma.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#  escrever os sinais filtrados em arquivos wav
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# verificar flag de escrita
if escrever_wav is True:
    # Amplitude de pico da amostra wav
    pico_wav = 2**(N_bits-1) - 1

    # normalizando os sinais para amplitude unitária
    esquerdo_pb_norm = sinal_esquerdo_pb/np.max(np.abs(sinal_esquerdo_pb))
    esquerdo_pa_norm = sinal_esquerdo_pa/np.max(np.abs(sinal_esquerdo_pa))
    esquerdo_soma_norm = sinal_esquerdo_soma/np.max(np.abs(sinal_esquerdo_soma))

    # Converter para int de 16 bits e escrever os arquivos wav dos sinais gerados
    wavfile.write('esquerdo_pb.wav', fs, np.int16(esquerdo_pb_norm*pico_wav))
    wavfile.write('esquerdo_pa.wav', fs, np.int16(esquerdo_pa_norm*pico_wav))
    wavfile.write('esquerdo_soma.wav', fs, np.int16(esquerdo_soma_norm*pico_wav))

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# usando ruído branco
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# criar vetor com sinal de ruído branco
ruido = np.random.randn(N_dft)

# aplicar os filtros
ruido_pb = ss.lfilter(b_passabaixas, a_passabaixas, ruido)
ruido_pa = ss.lfilter(b_passaaltas, a_passaaltas, ruido)

# calcular a DFT
ruido_f = np.fft.fft(ruido, N_dft)
ruido_pb_f = np.fft.fft(ruido_pb, N_dft)
ruido_pa_f = np.fft.fft(ruido_pa, N_dft)
ruido_pb_pa_f = np.fft.fft(ruido_pb + ruido_pa, N_dft)

# dividir o espectro de saída pelo espectro de entrada
razao_ruido_pb = ruido_pb_f/ruido_f
razao_ruido_pa = ruido_pa_f/ruido_f

razao_ruido_pb_pa = ruido_pb_pa_f/ruido_f

# plotar a razão entre os espectros de saída e entrada
plt.figure()
plt.subplot(211)
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(razao_ruido_pb[:N_dft//2+1])),
         label='Passa-baixas')
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(razao_ruido_pa[:N_dft//2+1])),
         color='C1', linestyle='--', label='Passa-altas')
plt.xlim(0, fs/2)
plt.ylim(-80, 20)
plt.legend(loc='center right')
plt.title('Razão dos espectros do ruído')
plt.ylabel('Magnitude [dB]')

plt.subplot(212)
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(razao_ruido_pb_pa[:N_dft//2+1])),
         color='C2', label='Passa-baixas + Passa-altas')
plt.xlim(0, fs/2)
plt.ylim(-80, 20)
plt.legend(loc='lower right')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz]')

if salvar_fig:
    plt.savefig('razao_espectros_ruido.png')


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Parte 2 - Eco
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- *-


def criar_eco(N_amostras, n_eco, amp_eco, fs):
    """
    Cria um sinal de eco com os parâmetros fornecidos
    
    Parâmetros:
    -----------
    N_amostras : int
        Número de amostras no sinal de eco
    n_eco : int
        Número de amostras de atraso para o eco
    amp_eco : float
        Amplitude do eco
    fs : float
        Frequência de amostragem
        
    Retorna:
    --------
    RI_eco : array
        Resposta ao impulso do eco
    """
    
    RI_eco = np.zeros(N_amostras)
    
    RI_eco[0] = 1.              # parte direta
    RI_eco[n_eco] = amp_eco     # parte do eco
    
    return RI_eco


# parâmetros do eco
N_amostras = 2**15
delta_t = 0.0455            # atraso do eco em segundos
amp_eco = 1                 # amplitude do eco
n_eco = int(np.round(delta_t*fs))   # tempo de eco em amostras (arredondado)

RI_eco1 = criar_eco(N_amostras, n_eco, amp_eco, fs)

t_eco = np.linspace(0, (N_amostras-1)/fs, N_amostras)

# plotar RI
plt.figure()
plt.stem(t_eco, RI_eco1)
plt.xlim([-0.005, 0.1])
plt.ylim([-0.2, 1.2])
plt.title('Resposta ao Impulso do Eco')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

if salvar_fig:
    plt.savefig('eco_RI.png')

# calcular e plotar a FRF do eco
FRF_eco = np.fft.fft(RI_eco1)

# FRF teórica do eco
FRF_teorica = 1 + amp_eco*np.exp(-1j*2*np.pi*f*n_eco/fs)

plt.figure()
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(FRF_eco[:N_amostras//2+1])),
         linewidth=3, label='Calculada')
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(FRF_teorica[:N_amostras//2+1])),
         linestyle='--', marker='^', label='Teórica')
plt.xlim([0, 150])
plt.ylim([-35, 10])
plt.title('Resposta em Frequência do Eco')
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend(loc='lower left')

if salvar_fig:
    plt.savefig('eco_FRF.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# aplicar eco ao sinal de música (usar amostras da RI como coeficientes do filtro)
# --->>> lembre de usar o sinal filtrado!
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

sinal_esquerdo_eco1 = ss.lfilter(RI_eco1, 1., sinal_esquerdo_soma)
sinal_e_eco1_f = np.fft.fft(sinal_esquerdo_eco1[:N_amostras])

# plotar os espectros do sinal
# -->>> note como o espectro do eco oscila em torno do espectro original
plt.figure(figsize=(12, 7))
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_esquerdo_f[:N_amostras//2+1])),
         label='Original')
plt.plot(f[:N_dft//2+1],
         20*np.log10(np.abs(sinal_e_eco1_f[:N_amostras//2+1])),
         '-.', label='com eco')
plt.xlim([0, 200])
plt.ylim([-20, 50])
plt.title('Espectro do sinal de música (com e sem eco)')
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend(loc='upper left')

if salvar_fig:
    plt.savefig('musica_eco.png')


# criar 2o eco, filtrar sinal de música com ele
delta_t2 = 0.001
n_eco2 = int(np.round(delta_t2*fs))     # tempo de eco em amostras (arredondado)
RI_eco2 = criar_eco(N_amostras, n_eco2, amp_eco, fs)

sinal_esquerdo_eco2 = ss.lfilter(RI_eco2, 1., sinal_esquerdo_soma)
sinal_e_eco2_f = np.fft.fft(sinal_esquerdo_eco2[:N_amostras])


# escrever arquivos wav
if escrever_wav is True:

    # Amplitude de pico da amostra wav
    pico_wav = 2**(N_bits-1) - 1

    # normalizando os sinais para amplitude unitária
    esquerdo_eco1_norm = sinal_esquerdo_eco1/np.max(np.abs(sinal_esquerdo_eco1))
    esquerdo_eco2_norm = sinal_esquerdo_eco2/np.max(np.abs(sinal_esquerdo_eco2))

    # Converter para int de 16 bits e escrever os arquivos wav dos sinais gerados
    wavfile.write('esquerdo_eco1.wav', fs, np.int16(esquerdo_eco1_norm*pico_wav))
    wavfile.write('esquerdo_eco2.wav', fs, np.int16(esquerdo_eco2_norm*pico_wav))


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Parte 2 (OPCIONAL) - Cenário fonte-parede
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- *-

fs = 44100.     # frequência de amostragem [Hz]
c0 = 343.       # velocidade do som [m/s]
N_dft = 2**15   # comprimento da DFT

t_eco = np.linspace(0, (N_dft-1)/fs, N_dft)

P1 = 0.1        # amplitude da fonte

R = 0.1         # raio da fonte
d_fp = 4.       # distância do centro da fonte até a parede
d_mp = 1.8      # distância do microfone até a parede

d_fm = d_fp-d_mp    # distância do centro da fonte até o microfone

# som direto
d_direto = d_fm - R                     # distância percorrida
t_direto = d_direto/c0                  # tempo de viagem [s]
n_direto = int(np.round(t_direto*fs))   # tempo de viagem [amostras]
a_direto = P1/d_direto                  # amplitude

# som refletido
d_refletido = d_direto + 2*d_mp                 # distância percorrida
t_refletido = d_refletido/c0                    # tempo de viagem [s]
n_refletido = int(np.round(t_refletido*fs))     # tempo de viagem [amostras]
a_refletido = P1/d_refletido                    # amplitude

RI_fonteparede = np.zeros(N_dft)
RI_fonteparede[n_direto] = a_direto          # parte direta
RI_fonteparede[n_refletido] = a_refletido    # parte refletida

# plotar RI
plt.figure()
plt.stem(t_eco, RI_fonteparede)
plt.xlim([-0.005, 0.025])
plt.ylim([-0.02, 0.08])
plt.title('Resposta ao Impulso Fonte-Parede')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

if salvar_fig:
    plt.savefig('fonte_parede_RI.png')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Parte 3 - OPCIONAL: Cálculo de RI e FRF de Filtros
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- *-

# Filtro 1 - filtro de média móvel, M=3
M1 = 3
RI_1 = np.zeros(15)
RI_1[:M1] = 1./M1

# calcular a FRF
w1, FRF1 = ss.freqz(RI_1, 1.)

plt.figure()
plt.stem(RI_1)
plt.title('RI 1 - Filtro MM, M=3')
plt.ylim([-0.5, 0.5])
plt.xlim([-1, 15])
plt.ylabel('Amplitude')
plt.xlabel('Amostras')

if salvar_fig:
    plt.savefig('Filtro1_RI.png')

plt.figure()
plt.subplot(211)
plt.plot(w1, 20*np.log10(np.abs(FRF1)))
plt.xlim([0, np.pi])
plt.ylabel('Magnitude [dB]')
plt.title('FRF 1 - Filtro MM, M=3')
plt.grid()

plt.subplot(212)
plt.plot(w1, np.angle(FRF1))
plt.xlim([0, np.pi])
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.ylabel('Ângulo [rad]')
plt.xlabel('Freq. Normalizada [rad/amostra]')

if salvar_fig:
    plt.savefig('Filtro1_FRF.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Filtro 2 - filtro de média móvel, M=5
M2 = 5
RI_2 = np.zeros(15)
RI_2[:M2] = 1./M2

# calcular a FRF
w2, FRF2 = ss.freqz(RI_2, 1.)

plt.figure()
plt.stem(RI_2)
plt.title('RI 2 - Filtro MM, M=5')
plt.ylim([-0.5, 0.5])
plt.xlim([-1, 15])
plt.ylabel('Amplitude')
plt.xlabel('Amostras')

if salvar_fig:
    plt.savefig('Filtro2_RI.png')

plt.figure()
plt.subplot(211)
plt.plot(w2, 20*np.log10(np.abs(FRF2)))
plt.xlim([0, np.pi])
plt.ylabel('Magnitude [dB]')
plt.title('FRF 2 - Filtro MM, M=5')
plt.grid()

plt.subplot(212)
plt.plot(w2, np.angle(FRF2))
plt.xlim([0, np.pi])
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.ylabel('Ângulo [rad]')
plt.xlabel('Freq. Normalizada [rad/amostra]')

if salvar_fig:
    plt.savefig('Filtro2_FRF.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Filtro 3 - filtro de média móvel, M=10
M3 = 10
RI_3 = np.zeros(15)
RI_3[:M3] = 1./M3

# calcular a FRF
w3, FRF3 = ss.freqz(RI_3, 1.)

plt.figure()
plt.stem(RI_3)
plt.title('RI 3 - Filtro MM, M=10')
plt.ylim([-0.5, 0.5])
plt.xlim([-1, 15])
plt.ylabel('Amplitude')
plt.xlabel('Amostras')

if salvar_fig:
    plt.savefig('Filtro3_RI.png')

plt.figure()
plt.subplot(211)
plt.plot(w3, 20*np.log10(np.abs(FRF3)))
plt.xlim([0, np.pi])
plt.ylabel('Magnitude [dB]')
plt.title('FRF 3 - Filtro MM, M=10')
plt.grid()

plt.subplot(212)
plt.plot(w3, np.angle(FRF3))
plt.xlim([0, np.pi])
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.ylabel('Ângulo [rad]')
plt.xlabel('Freq. Normalizada [rad/amostra]')

if salvar_fig:
    plt.savefig('Filtro3_FRF.png')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Filtro 4 - filtro de diferença
RI_4 = np.zeros(15)
RI_4[0] = 1.
RI_4[1] = -1

# calcular a FRF
w4, FRF4 = ss.freqz(RI_4, 1.)

plt.figure()
plt.stem(RI_4)
plt.title('RI 4 - Filtro de Diferença')
plt.ylim([-1.5, 1.5])
plt.xlim([-1, 15])
plt.ylabel('Amplitude')
plt.xlabel('Amostras')

if salvar_fig:
    plt.savefig('Filtro4_RI.png')

plt.figure()
plt.subplot(211)
plt.plot(w4, 20*np.log10(np.abs(FRF4)))
plt.xlim([0, np.pi])
plt.ylabel('Magnitude [dB]')
plt.title('FRF 4 - Filtro de Diferença')
plt.grid()

plt.subplot(212)
plt.plot(w4, np.angle(FRF4))
plt.xlim([0, np.pi])
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.ylabel('Ângulo [rad]')
plt.xlabel('Freq. Normalizada [rad/amostra]')

if salvar_fig:
    plt.savefig('Filtro4_FRF.png')