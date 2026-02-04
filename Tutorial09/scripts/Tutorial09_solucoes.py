"""
Exemplo de solução para o Tutorial 09 de Processamento Digital de Sinais

https://github.com/fchirono/AulasDSP

Resposta ao Impulso da sala acessado de:
    Stewart, Rebecca and Sandler, Mark. "Database of Omnidirectional and 
    B-Format Impulse Responses", in Proc. of IEEE Int. Conf. on Acoustics, 
    Speech, and Signal Processing (ICASSP 2010), Dallas, Texas, March 2010.
    URL: http://isophonics.net/content/room-impulse-response-data-set

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
from scipy.io import wavfile

import sounddevice as sd

import matplotlib.pyplot as plt
plt.close('all')

# flag para salvar figuras
save_figs = False


# %% Parte 1 - Convolução

#def convtempo(x, h):
#    # implementa a convolução de sinais de tempo discreto
#    # no domínio do tempo
#    # --------------------------
#    L = x.shape[0]
#    M = h.shape[0]
#
#    Q = L + M - 1
#    y = np.zeros(Q)
#
#    for n in range(Q):
#        for m in range(M):
#            idx_atual = n - m
#            if (idx_atual >= 0) and (idx_atual < L):
#                y[n] += h[m]*x[idx_atual]
#    return y

def convtempo(x, h):
    # implementa a convolução de sinais de tempo discreto
    # no domínio do tempo com alta eficiência
    # --------------------------
    L = x.shape[0]  # obter comprimento de x
    M = h.shape[0]  # obter comprimento de h
    Q = L + M - 1   # calcular comprimento do resultado da convolução
    y = np.zeros(Q)     # alocar memória para o resultado

    for n in range(Q):   # para todos os valores do resultado
        # encontrando os intervalos entre os quais ler x e h, respectivamente
        m_max = np.min([n+1, M])     # índice superior de h para soma atual
        m_min = np.max([0, n+1-L])   # índice inferior de h para soma atual

        xmax = np.min([n+1, L])     # índice superior de x para soma atual
        xmin = np.max([0, n-M+1])   # índice inferior de x para soma atual

        y[n] = np.dot(h[m_min:m_max], np.flipud(x[xmin:xmax]))
    return y


def convdft(x, h):
    # implementa a convolução de sinais de tempo discreto
    # no domínio da frequência
    # --------------------------
    L = x.shape[0]
    M = h.shape[0]

    Q = L + M - 1

    X = np.fft.fft(x, Q)  # realizar FFT com o comprimento do sinal final
    H = np.fft.fft(h, Q)  # realizar FFT com o comprimento do sinal final

    Y = X*H  # realizar convolução no domínio da frequência

    y = np.real(np.fft.ifft(Y))
    return y


# %%
x = np.array([3, 4, 5])     # gerar primeiro sinal
h = np.array([1, 2, 3, 4])  # gerar segundo sinal

y_tempo = convtempo(x, h)     # usar a função do domínio do tempo
y_dft = convdft(x, h)         # usar a função do domínio da frequência
y_pyth = np.convolve(x, h)    # usar a função do numpy

n = np.arange(0, x.shape[0] + h.shape[0]-1, 1)    # vetor de amostras

plt.figure()
# retorna identificadores do gráfico de hastes
(marcador_tempo, haste_tempo, base_tempo) = plt.stem(n, y_tempo, label='y_tempo')

# altera propriedades do gráfico de hastes através dos identificadores
plt.setp(marcador_tempo, markersize=15, marker='o')
plt.setp(haste_tempo, color='green', linewidth=2, linestyle='dashed')
plt.setp(base_tempo, color='red', linewidth=2)

# aplica mudanças similares aos outros gráficos
(marcador_dft, haste_dft, _) = plt.stem(n, y_dft, label='y_DFT')
plt.setp(marcador_dft, markersize=12, color='r', marker='s')
plt.setp(haste_dft, color='r', linewidth=2, linestyle='dashdot')

(marcador_py, haste_py, _) = plt.stem(n, y_pyth, label='y_python')
plt.setp(marcador_py, markersize=10, color='g', marker='^')
plt.setp(haste_py, color='g', linewidth=2, linestyle='dashdot')


plt.grid()
plt.legend(loc='upper left')
plt.xlim(-1, x.shape[0]+h.shape[0]-1)
plt.ylim(-5, 40)
plt.title('Resultados da Convolução')
plt.xlabel('Índice da Amostra')
plt.ylabel('Valor')
if save_figs: 
    plt.savefig('VariantesConv.png')


# %% Parte 2 - Aplicações da Convolução

# ler arquivos wav e sua frequência de amostragem
fs_IR, IR_16bits = wavfile.read("C4DM_GreatHall_Omni_x00y05.wav")
fs_Voz, Voz_16bits = wavfile.read("voz.wav")

# confirmar o tipo de dado obtido do arquivo wav
IR_16bits.dtype

# Contabilizar um arquivo de 16 bits
N_bits = 16

# normalização dos dados de áudio de 16 bits
IR = IR_16bits/(2.**(N_bits - 1)-1)
Voz = Voz_16bits/(2.**(N_bits - 1)-1)

# Plotar a IR (Resposta Impulsiva)
plt.figure()
t = np.linspace(0, IR.shape[0]/fs_IR, IR.shape[0])
plt.plot(t, IR)
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('Resposta ao Impulso')
plt.grid()
if save_figs: 
    plt.savefig('RespImpulso.png')
    
# Realizar convolução
Voz_dft = convdft(Voz, IR)
Voz_pyth = np.convolve(Voz, IR)

# Amplitude de pico da amostra wav
pico_wav = 2**(N_bits-1) - 1

# normalizando os sinais para amplitude unitária
Voz_dft_norm = Voz_dft/np.max(np.abs(Voz_dft))
Voz_pyth_norm = Voz_pyth/np.max(np.abs(Voz_pyth))

# # Auraliza o sinal original e o sinal processado
# sd.play(Voz_16bits, fs_Voz, blocking=True)
# sd.play(Voz_dft, fs_Voz, blocking=True)

# Converter para inteiro de 16 bits e escrever os arquivos wav dos sinais gerados
#wavfile.write('voz_dft.wav', fs_Voz, np.int16(Voz_dft_norm*pico_wav))
#wavfile.write('voz_pyth.wav', fs_Voz, np.int16(Voz_pyth_norm*pico_wav))

# %% OPCIONAL: Parte 3 - Convolução Overlap-Add (Sobreposição-Adição)

L = 100         # número de amostras no vetor

x = np.ones(L)  # gerar vetor de comprimento L=100 apenas com uns

h = np.array([1, 1, 1, 1])  # gerar vetor de resposta impulsiva

Q = h.shape[0]  # obter comprimento de h

N = 10                      # comprimento dos quadros
M = int(np.ceil(L/N))       # obter número de quadros e converter para inteiro
X_10 = np.zeros([M, N])     # alocar memória para a matriz de quadros

# rearranjando x em X
for n in range(M):
    X_10[n, :] = x[n*N:(n+1)*N]  # ler quadros de comprimento N

K = N + Q - 1   # calcular comprimento da convolução de cada quadro com h

Y_10 = np.zeros([M, K])    # alocar memória para resultado da convolução

for n in range(M):
    # convoluir quadros em X com h e salvar em Y
    Y_10[n, :] = convtempo(X_10[n, :], h)

y_10 = np.zeros(L + Q - 1)  # alocar memória para o resultado final

for n in range(M):
    # atenção às extremidades sobrepostas e portanto usar '+='!
    y_10[n*N:n*N+K] += Y_10[n, :]

# gerar sinal de saída sem overlap-add para comparação
y_teste = np.convolve(x, h)

# Plotar o resultado da operação de convolução
plt.figure()
plt.stem(y_10)
plt.grid()
plt.xlim(-5, y_10.shape[0]+5)
plt.ylim(-1, 5)
plt.title('Resultado Overlap-Add')
plt.xlabel('Índice da Amostra')
plt.ylabel('Valor')
if save_figs: 
    plt.savefig('OverlapAdd.png')

# Plotar diferença entre os dois resultados
plt.figure()
plt.plot(y_10-y_teste)   # plotar diferença entre ambos os resultados

# "somando" strings para manter comprimento da linha dentro dos limites
plt.title('Diferença entre o resultado Overlap-Add e o resultado da ' +
          'convolução simples')

# ----- Mudar para N = 52 ------------

N = 52                      # definir novo comprimento de quadro
M = int(np.ceil(float(L)/N))       # obter número de quadros e converter para inteiro
X_52 = np.zeros([M, N])     # alocar memória para a matriz de quadros

# rearranjando x em X
for n in range(M):
    if ((n+1)*N <= x.shape[0]):   # quadro atual pode ser lido com comprimento completo?
        X_52[n, :] = x[n*N:(n+1)*N]  # ler quadros de comprimento N
    else:
        R = x.shape[0] - n*N  # calcular comprimento do quadro final
        X_52[n, 0:R] = x[n*N:]  # ler o restante

K = N + Q - 1   # calcular comprimento da convolução de cada quadro com h

Y_52 = np.zeros([M, K])    # alocar memória para resultado da convolução

for n in range(M):
    # convoluir quadros em X com h e salvar em Y
    Y_52[n, :] = convtempo(X_52[n, :], h)


y_52 = np.zeros(L + Q - 1)  # alocar memória para o resultado final

for n in range(M):
    # quadro atual pode ser escrito com comprimento completo?
    if ((n*N+K) <= y_52.shape[0]):
        # atenção às extremidades sobrepostas e portanto usar '+='!
        y_52[n*N:n*N+K] += Y_52[n, :]
    else:
        R = y_52.shape[0] - n*N     # calcular comprimento do quadro final
        y_52[n*N:] += Y_52[n, 0:R]   # escrever o restante

# Plotar diferença entre os dois resultados
plt.figure()
plt.plot(y_52-y_teste)   # plotar diferença entre ambos os resultados
plt.title('Diferença entre o método Overlap-Add e o resultado da ' +
            'convolução simples')


def OverlapAdd(x, h, N_DFT):
    # realiza a convolução de x e h usando o método Overlap-Add
    # -----------------------

    L = x.shape[0]
    Q = h.shape[0]
    M = int(np.ceil(float(L)/N_DFT))   # obter número de quadros e converter para inteiro
    y = np.zeros(L+Q-1)         # alocar memória para o resultado final
    K = N_DFT + Q - 1       # calcular comprimento da convolução de cada quadro com h

    # rearranjando x em X
    for n in range(M):

        # quadro atual pode ser lido com comprimento completo?
        if ((n+1)*N_DFT <= x.shape[0]):

            # convoluir quadros em X com h e salvar em Y
            y[n*N_DFT:n*N_DFT+K] += np.convolve(x[n*N_DFT:(n+1)*N_DFT], h)

        else:
            # convoluir o restante
            y[n*N_DFT:] += np.convolve(x[n*N_DFT:], h)

    return y


y_OA = OverlapAdd(x, h, 12)

# Plotar diferença entre os dois resultados
plt.figure()
plt.plot(y_OA-y_teste)   # plotar diferença entre ambos os resultados
plt.title('Diferença entre a função Overlap-Add e o resultado da ' +
            'convolução simples')