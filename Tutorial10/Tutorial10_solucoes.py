"""
Exemplo de solução para o Tutorial 10 de Processamento Digital de Sinais

https://github.com/fchirono/AulasDSP

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
from scipy import signal as sig
from scipy import io

import matplotlib.pyplot as plt
plt.close('all')


# flag para salvar figuras
save_figs = False


# %% SEÇÃO: Definições de Funções


def esp_cruzado(x, y, win=None, N_dft=4096, N_sobreposicao=0):
    """
    Calcula o espectro de potência cruzado das duas sequências em tempo discreto 'x'
    e 'y'. Este código usa o método de Welch, e permite a seleção da
    função de janelamento e quantidade de sobreposição.

    O mesmo código permite o cálculo do auto-espectro de potência ao
    chamá-lo como "esp_cruzado(x, x)".
    """

    # 'Truque' para definir a janela padrão como uma Hann de comprimento 'N_dft'
    if win is None:
        win = sig.hann(N_dft)

    # calcular normalização da janela
    U = np.sum(win**2)/N_dft

    # encontrar o número mínimo de amostras
    # -> transformar em float para permitir a operação 'ceil' abaixo
    K = np.max([x.shape[0], y.shape[0]])

    # calcular o número de quadros
    L = int(np.ceil((float(K) - N_sobreposicao)/(N_dft - N_sobreposicao)))

    # deslocamento entre quadros
    D = N_dft - N_sobreposicao

    # fazer o zero-padding necessário
    xNz = int(L*N_dft - x.shape[0])
    yNz = int(L*N_dft - y.shape[0])
    x = np.concatenate((x, np.zeros(xNz)))
    y = np.concatenate((y, np.zeros(yNz)))

    # alocar memória para saída
    Pxy = np.zeros(N_dft, 'complex')

    # Para cada quadro...
    for l in range(L):

        # Calcular as DFTs de ambos os sinais (do quadro atual)
        X = np.fft.fft(win*x[(D*l):(D*l + N_dft)])
        Y = np.fft.fft(win*y[(D*l):(D*l + N_dft)])

        # Adicionar a estimação atual do espectro de potência cruzado à soma geral
        Pxy += X*Y.conj()/(N_dft*U)

    # retornar a estimação média do espectro de potência cruzado
    return Pxy/L


def est_sistema(x, y, win, N_dft, N_sobreposicao=0):
    """
    Estima a resposta ao impulso 'h' e a resposta em frequência 'H' de um sistema
    a partir de um sinal de entrada 'x' e do sinal de saída do sistema 'y' usando
    o Estimador H1.

    Este estimador calcula o espectro de potência cruzado entre entrada e saída
    e o auto-espectro de potência da entrada; a FRF do sistema é então estimada
    a partir da divisão do espectro de potência cruzado entrada-saída pelo
    auto-espectro de potência da entrada.

    A resposta ao impulso é então calculada a partir da FRF estimada usando uma
    FFT Inversa.
    """

    # Calcula o espectro de potência cruzado entrada-saída
    Pyx = esp_cruzado(y, x, win, N_dft, N_sobreposicao)

    # Calcula o auto-espectro de potência da entrada
    Pxx = esp_cruzado(x, x, win, N_dft, N_sobreposicao)

    # Estima a FRF usando o Estimador H1
    H = Pyx/Pxx

    # Calcula a IFFT da FRF para obter a resposta ao impulso
    h = np.real(np.fft.ifft(H))

    return h, H


def coerencia(x, y, win, N_dft, N_sobreposicao):
    """
    Calcula a função de coerência ao quadrado entre dois sinais usando seus
    auto-espectros de potência e seu espectro de potência cruzado.

    A função de coerência 'gama' é de valor real por definição, mas o
    cálculo pode introduzir artefatos numéricos na forma de pequenas
    partes imaginárias nos valores de coerência. Estes artefatos são tratados
    mantendo apenas a parte real do cálculo e descartando as partes
    imaginárias.
    """

    # Calcula o auto-espectro de potência do 1o sinal
    Pxx = esp_cruzado(x, x, win, N_dft, N_sobreposicao)

    # Calcula o auto-espectro de potência do 2o sinal
    Pyy = esp_cruzado(y, y, win, N_dft, N_sobreposicao)

    # Calcula o espectro de potência cruzado entre os dois sinais
    Pyx = esp_cruzado(y, x, win, N_dft, N_sobreposicao)

    # Calcula a função de coerência 'gama^2', descartando a parte imaginária
    gama_quadrado = np.real(np.abs(Pyx)**2/Pxx/Pyy)

    return gama_quadrado


# %% Tarefa 1

# carregar arquivo MAT como um dicionário
arquivo_mat = io.loadmat('tutorial10')

# ler a chave da resposta em frequência do dicionário
H_sistema = arquivo_mat['H_sistema']
h_sistema = arquivo_mat['h_sistema']
N_dft = arquivo_mat['N_dft'][0, 0]

# mudar de array 2D para vetor (1D)
H_sistema = H_sistema[0, :]
h_sistema = h_sistema[0, :]

# Definir variáveis auxiliares
fs = 44100                             # frequência de amostragem
df = float(fs)/N_dft                   # incremento de frequência
f = np.linspace(0, fs-df, N_dft)       # vetor de frequência

janela = sig.windows.hann(N_dft)
overlap = 0

# criar figura para verificar a resposta em frequência
# (mais conteúdo é adicionado a ela posteriormente)
plt.figure(1)
plt.semilogx(f[0:int(N_dft/2+1)], 20*np.log10(np.abs(H_sistema)), 'b--',
             linewidth=4, label='Original')
plt.xlim(10, fs/2)
#plt.ylim(-60, 5)
plt.xlabel('Frequência [Hz]')
plt.ylabel('FRF [dB]')
plt.title('Resposta em Frequência do Sistema')
plt.grid()

if save_figs:
    plt.savefig('RespFreqT1.png')

# %% Tarefa 2

# reconstruir vetor FFT completo
H_sistema_rec = np.zeros(N_dft, 'complex')
H_sistema_rec[0:int(N_dft/2)+1] = H_sistema
H_sistema_rec[int(N_dft/2)+1:] = H_sistema[int(N_dft/2)-1:0:-1].conj()

# aplicar ifft simétrica para obter a resposta ao impulso
h_sistema1 = np.real(np.fft.ifft(H_sistema_rec))[:11]
#   --> a saída da IFFT retorna N_fft amostras; apenas as primeiras 11 amostras são
#       IR real, o resto apenas adiciona ruído e reduz a coerência sobre a
#       banda de frequência de interesse!

# verificar a resposta ao impulso
plt.figure(2)
plt.subplot(211)    # comentar o comando subplot para gerar a fig 'RI_T2.png'

# retornar handles de parâmetros do gráfico de hastes
(marker_rec, stem_rec, base_rec) = plt.stem(h_sistema1[0:11],
                                            label='IR Reconstruída')
# alterar propriedades do gráfico de hastes através dos handles de parâmetros
plt.setp(marker_rec, markersize=12, marker='o')
plt.setp(stem_rec, color='green', linewidth=2, linestyle='dashed')
plt.setp(base_rec, color='red', linewidth=2)

(marker_orig, stem_orig, base_orig) = plt.stem(h_sistema, 'r', label='IR Original')
plt.setp(marker_orig, markersize=8, color='r', marker='s')
plt.setp(stem_orig, color='r', linewidth=2, linestyle='dashdot')

plt.ylabel('Amplitude')
plt.xlabel('Índice da Amostra')
plt.title('Resposta ao Impulso Original')
plt.xlim(-1, 11)
plt.ylim(-0.1, 0.4)
plt.legend()
plt.grid()

if save_figs:
    plt.savefig('RI_T2.png')

# %% Tarefa 3

# número de amostras
K = 44100

# criar vetor de ruído
x = np.random.randn(K)

# calcular convolução de 'x' e 'h_sistema1'
y = np.convolve(x, h_sistema1)

# %% Tarefa 5

# Estimar o Espectro de Potência de 'x' usando valores padrão para 'win', 'N_dft'
# e 'N_sobreposicao'
Pxx = esp_cruzado(x, x, janela, N_dft, overlap)

# Estimar o Espectro de Potência de 'y' com valores escolhidos para 'win', 'N_dft' e
# 'N_sobreposicao'
Pyy = esp_cruzado(y, y, janela, N_dft, overlap)

# Estimar o Espectro Cruzado de 'x' e 'y' com valores escolhidos
Pyx = esp_cruzado(y, x, janela, N_dft, overlap)

# Calcular a função de coerência ao quadrado, descartando qualquer parte imaginária
gama_quadrado = np.real( (Pyx * Pyx.conj()) / (Pxx * Pyy))

plt.figure(3)
plt.semilogx(f[:int(N_dft/2+1)], gama_quadrado[:int(N_dft/2+1)])
plt.xlim(10, fs/2)
plt.ylim([0, 1.1])
plt.xlabel('Frequência [Hz]')
plt.ylabel('Coerência')
plt.title('Coerência entre sinal de entrada e saída')
plt.grid()

if save_figs:
    plt.savefig('Coerencia_T5.png')

# %% Tarefa 6

# Estimar a IR e FRF do sistema a partir de sua entrada e saída
[h_sistema2, H_sistema2] = est_sistema(x, y, janela, N_dft, overlap)

plt.figure(2)
plt.subplot(212)
plt.stem(h_sistema2[0:11], label='IR Estimada')
plt.xlabel('Índice da amostra')
plt.ylabel('Amplitude')
plt.title('Resposta ao Impulso Estimada')
plt.xlim(-1, 11)
plt.ylim(-0.1, 0.4)
plt.grid()
plt.tight_layout()

if save_figs:
    plt.savefig('RespImpT6.png')


# Adicionar FRF estimada à 1a figura
plt.figure(1)
plt.semilogx(f[:int(N_dft/2+1)], 20*np.log10(np.abs(H_sistema2[:int(N_dft/2+1)])),
             'r:', linewidth=4, label='Estimada')
plt.legend(loc='lower left')
if save_figs:
    plt.savefig('RespFreqT6.png')

# %% Tarefa 7

# gerar ruído branco
n = np.random.randn(y.shape[0])

# fator de ganho para ruído
alpha = 0.5

# calcular nova saída ruidosa
y_tilde = y + alpha*n

# estimar o sistema a partir da saída ruidosa
[h_sistema3, H_sistema3] = est_sistema(x, y_tilde, janela, N_dft, overlap)

# Plotar IR estimada
plt.figure(8)
plt.stem(h_sistema3[0:11])
plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.title('IR Estimada de Dados Ruidosos')
plt.xlim(-1, 11)
plt.ylim(-0.1, 0.4)
plt.grid()

if save_figs:
    plt.savefig('RespImpT7.png')


# Plotar erro relativo entre IRs estimadas e original
plt.figure(9)

(marker_clean, stem_clean, _) = plt.stem(h_sistema2[0:11] - h_sistema,
                                         label='Est. Limpa')
plt.setp(marker_clean, markersize=14, marker='o')
plt.setp(stem_clean, color='blue', linewidth=2, linestyle='dashed')

(marker_noisy, stem_noisy, _) = plt.stem(h_sistema3[0:11] - h_sistema,
                                         label='Est. Ruidosa')
plt.setp(marker_noisy, markersize=10, color='r', marker='s')
plt.setp(stem_noisy, color='r', linewidth=2, linestyle='dashdot')

plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.title('Erro na Estimação da IR')
plt.xlim(-1, 11)
plt.ylim(-0.04, 0.01)
plt.grid()
plt.legend(loc='lower right')
if save_figs:
    plt.savefig('ErroEstimacaoT7.png')


# Plotar FRF do sistema estimada a partir da saída ruidosa
plt.figure(1)
plt.semilogx(f[:int(N_dft/2+1)], 20*np.log10(np.abs(H_sistema3[:int(N_dft/2+1)])),
             'g', label='Est. (saída ruidosa)')
plt.legend(loc='lower left')
if save_figs:
    plt.savefig('RespFreqT7.png')

# Plotar coerência entre entrada e saída ruidosa
gama_quadrado_x_ytilde = coerencia(x, y_tilde, janela, N_dft, overlap)

plt.figure(4)
plt.semilogx(f[:int(N_dft/2+1)], gama_quadrado_x_ytilde[:int(N_dft/2+1)])
plt.xlim(10, fs/2)
plt.xlabel('Frequência [Hz]')
plt.ylabel('Coerência')
plt.title('Coerência entre entrada e saída ruidosa')
plt.grid()
if save_figs:
    plt.savefig('Coerencia_T7.png')

# %% Tarefa 8
K = 8192
h_8a = np.zeros(K)
h_8a[200] = 1.

h_8b = np.zeros(K)
h_8b[1000] = 1.

h_8c = np.zeros(K)
h_8c[6000] = 1.

y_8a = np.convolve(x, h_8a)
y_8b = np.convolve(x, h_8b)
y_8c = np.convolve(x, h_8c)

# Estimar a IR e FRF do sistema a partir de sua entrada e saída
[h_est8a, H_est8a] = est_sistema(x, y_8a, janela, N_dft, overlap)
[h_est8b, H_est8b] = est_sistema(x, y_8b, janela, N_dft, overlap)
[h_est8c, H_est8c] = est_sistema(x, y_8c, janela, N_dft, overlap)

# Plotar FRFs estimadas
plt.figure(6)
plt.semilogx(f[:int(N_dft/2+1)], 20*np.log10(np.abs(H_est8a[:int(N_dft/2+1)])),
             ':', linewidth=2, label='M=200')
plt.semilogx(f[:int(N_dft/2+1)], 20*np.log10(np.abs(H_est8b[:int(N_dft/2+1)])),
             '--', label='M=1000')
plt.semilogx(f[:int(N_dft/2+1)], 20*np.log10(np.abs(H_est8c[:int(N_dft/2+1)])),
             '-', label='M=6000')
plt.legend(loc='lower left')
plt.title('Estimações de FRF')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequência [Hz}')
plt.xlim(10, fs/2)
plt.grid()
if save_figs:
    plt.savefig('RespFreqT8.png')


# Calcular e plotar função de coerência da estimação FRF do sistema
cohere_8a = coerencia(x, y_8a, janela, N_dft, overlap)
cohere_8b = coerencia(x, y_8b, janela, N_dft, overlap)
cohere_8c = coerencia(x, y_8c, janela, N_dft, overlap)


plt.figure(7)
plt.semilogx(f[:int(N_dft/2+1)], cohere_8a[:int(N_dft/2+1)], ':', linewidth=2,
             label='M=200')
plt.semilogx(f[:int(N_dft/2+1)], cohere_8b[:int(N_dft/2+1)], '--',
             label='M=1000')
plt.semilogx(f[:int(N_dft/2+1)], cohere_8c[:int(N_dft/2+1)], '-',
             label='M=6000')
plt.xlabel('Frequência [Hz]')
plt.ylabel('Coerência')
plt.title('Coerência entre os sinais de entrada e saída')
plt.legend(loc='center right')
plt.xlim(10, fs/2)
plt.grid()
if save_figs:
    plt.savefig('Coerencia_T8.png')