"""
Exemplo de solucao para o Tutorial 5 de Processamento Digital de Sinais

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
from matplotlib import pyplot as plt

import sounddevice as sd

import tutorial2_funcoes as Tutorial2


def sintetizar(f0, fs, T_max, A_k):
    """
    Uma funcao que implementa a Equacao (1):

        y = sintetizar(f0, fs, T_max, VetorAmpHarm)

    onde f0 eh a frequencia fundamental, fs eh a frequencia de amostragem, 
    T_max eh a duracao do sinal (em segundos), e A_k eh o vetor [a1, ... , aK]
    que contem as amplitudes dos harmonicos superiores. O primeiro elemento
    refere-se a amplitude da frequencia fundamental.

    Por exemplo, para gerar um sinal amostrado a 44100Hz, frequência fundamental
    A0 (440Hz), duracao 1s, com tres harmonicos de amplitude [1, 0.4, 0.2],
    a chamada da funcao eh:
    
        y = sintetizar(440, 44100, 1, [1.0, 0.4, 0.2])
    """

    # cria vetor de tempo
    t = np.linspace(0, T_max, int(T_max*fs), endpoint=False)

    # cria array de zeros
    y = np.zeros(t.shape)

    for k in range(A_k.shape[0]):
        y += A_k[k]*np.cos(2*np.pi*(k+1)*f0*t)

    return y, t


def AnaliseFourier(x, K):
    """ Calcula os coeficientes da serie de Fourier dos dados 'x' ate a
    ordem 'K' """

    ck = np.zeros(2*K+1, 'complex')    # aloca um vetor de zeros
    N = x.shape[0]

    for k in range(-int(K), int(K)+1):
        ck[k + K] = (1./N)*np.dot(x, np.exp(-1j*k*2*np.pi*np.arange(N)/N))

    return ck


def SinteseFourier(ck, N):
    """ Sintetiza um sinal periodico baseado em um vetor de coeficientes
    de Fourier; uma sintese de ordem "K" requer 2K+1 coeficientes.
    """

    n = np.arange(N)

    x = np.zeros(phi.shape, 'complex')
    K = (ck.size-1)//2

    for k in range(-int(K), int(K)+1):
        x += ck[k+K]*np.exp(1j*k*2*np.pi*n/N)

    return x


plt.close('all')


# %% *-*-* Tarefa 1 - Implementar a funcao e sintetizar uma nota A4 (440 Hz) *-*-*

# Define os parametros da nota
fs = 44100                             # frequencia de amostragem [Hz]
f0 = 440                               # Frequencia da nota [Hz]
T_max = 1.                             # duracao da nota [s]
amplitudes = np.array([1., 0.4, 0.2])  # amplitudes dos harmonicos

# Sintetiza o sinal e os vetores de tempo usando a funcao criada pelo usuario
sinal, tempo = sintetizar(f0, fs, T_max, amplitudes)

# Plota as primeiras amostras do sinal sintetizado
plt.figure()
plt.plot(tempo[:500], sinal[:500])
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('Nota A4')
plt.xlim([0, tempo[500]])
# plt.savefig("A4.png")

# # auraliza o sinal
# sd.play(sinal, fs, blocking=True)

# %% *-*-*-*-*-*-*-*-* Tarefa 2 - Sintetizar a melodia *-*-*-*-*-*-*-*-*-*-*-*-*

# define as amplitudes dos harmonicos
harmonicos = np.array([1., 1., 1., 1.])
# harmonicos = np.array([0.8, 0.1, 0.2, 0.6])

# Gera as notas
ds4, _ = sintetizar(311.13, fs, 0.15, harmonicos)
e4, _ = sintetizar(329.63, fs, 0.15, harmonicos)
f4, _ = sintetizar(349.23, fs, 0.15, harmonicos)
gs4, _ = sintetizar(415.30, fs, 0.15, harmonicos)
a4, _ = sintetizar(440.0, fs, 0.15, harmonicos)
silencio = np.zeros(int(0.15*fs))

# Gera a melodia concatenando as notas
melodia = np.copy(e4)
melodia = np.concatenate((melodia, ds4))
melodia = np.concatenate((melodia, e4))
melodia = np.concatenate((melodia, f4))
melodia = np.concatenate((melodia, e4))
melodia = np.concatenate((melodia, silencio))
melodia = np.concatenate((melodia, a4))
melodia = np.concatenate((melodia, silencio))
melodia = np.concatenate((melodia, e4))
melodia = np.concatenate((melodia, ds4))
melodia = np.concatenate((melodia, e4))
melodia = np.concatenate((melodia, f4))
melodia = np.concatenate((melodia, e4))
melodia = np.concatenate((melodia, silencio))
melodia = np.concatenate((melodia, gs4))
melodia = np.concatenate((melodia, silencio))

# # auraliza a melodia
# sd.play(0.5*melodia, fs, blocking=True)

# %% Parte 2 - plotando os padroes de radiação

# cria angulos entre 0 e 2*pi radianos
phi = np.linspace(0, 2*np.pi, 360, endpoint=False)

# Obtem os dois padroes de radiacao
padrao_rad1 = Tutorial2.PadraoRadiacao1(phi)
padrao_rad2 = Tutorial2.PadraoRadiacao2(phi)

# plota as figuras conforme as instrucoes do tutorial
FigPolar1 = plt.figure(figsize=(7, 7))
EixoPolar1 = FigPolar1.add_subplot(111, polar=True)
EixoPolar1.set_theta_zero_location("N")
EixoPolar1.plot(phi, np.abs(padrao_rad1))
EixoPolar1.set_title("Padrao de Radiacao 1", size=18)
# plt.savefig("PadraoRad1.png")

FigPolar2 = plt.figure(figsize=(7, 7))
EixoPolar2 = FigPolar2.add_subplot(111, polar=True)
EixoPolar2.set_theta_zero_location("N")
EixoPolar2.plot(phi, np.abs(padrao_rad2))
EixoPolar2.set_title("Padrao de Radiacao 2", size=18)
# plt.savefig("PadraoRad2.png")

# %%

"""
Esta proxima parte mostra como usar o metodo orientado a objetos para plotar
figuras; este metodo, embora mais complicado, permite um grau muito maior de
controle sobre o grafico.

Este nivel de detalhe nao eh exigido para estes tutoriais, estamos apenas
demonstrando o grau de controle que o Matplotlib lhe permite ter!
"""

# Cria uma nova figura com um tamanho predefinido
fig_polar1 = plt.figure(figsize=(7, 7))

# Adiciona um unico "subplot" a figura, usando eixos polares
ax_polar1 = fig_polar1.add_subplot(111, polar=True)

# Plota os dados no eixo polar
plot_polar1 = ax_polar1.plot(phi, np.abs(padrao_rad1))

# Adiciona um titulo ao subplot e muda o tamanho da fonte do titulo
titulo_polar1 = ax_polar1.set_title('Padrao de Radiacao 1', size=18)

# Move o titulo ligeiramente para cima, para que nao se sobreponha ao
# indicador de 90 graus
titulo_polar1.set_y(1.09)

# Ativa o 'layout apertado' para remover alguns espacos vazios fora do
# subplot (tente desativa-lo para observar a diferenca)
fig_polar1.tight_layout()


# Igual a figura anterior - veja acima para detalhes
fig_polar2 = plt.figure(figsize=(7, 7))
ax_polar2 = fig_polar2.add_subplot(111, polar=True)
plot_polar2 = ax_polar2.plot(phi, np.abs(padrao_rad2))
titulo_polar2 = ax_polar2.set_title('Padrao de Radiacao 2', size=18)

# Define o raio maximo para o grafico polar
ax_polar2.set_rmax(1)

# Define a posicao 'zero graus' como 'Norte' (ou seja, apontando para cima)
ax_polar2.set_theta_zero_location('N')

# Define o angulo ('theta') para incrementar na direcao horaria
ax_polar2.set_theta_direction('clockwise')

# Define as localizacoes das marcacoes de grade na direcao radial e adiciona
# rotulos a elas. Note que os pontos de grade devem ser estritamente
# positivos, e eh necessario adicionar um pequeno raio positivo se voce
# quiser visualizar o marcador zero.
ax_polar2.set_rgrids([0.0001, 0.2, 0.4, 0.6, 0.8, 1.0],
                     labels=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                     angle=-88)

# Define as localizacoes das marcacoes de grade na direcao 'theta' e adiciona
# rotulos em radianos. Esta notacao usa sintaxe LaTeX para exibir
# simbolos matematicos nas marcacoes de grade; nao se preocupe caso voce nao
# conheca LaTeX, o resultado eh puramente estetico.
#
# O comando 'frac=1.1' diz para adicionar os rotulos um pouco mais longe dos
# eixos do que o padrao, para fins de legibilidade (o padrao eh 'frac=1').
ax_polar2.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                         labels=[r'$\theta = 0$', r'$+\frac{\pi}{4}$',
                                 r'$+\frac{\pi}{2}$', r'$+\frac{3\pi}{4}$',
                                 r'$\pm \pi$', r'$-\frac{3\pi}{4}$',
                                 r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$'],
                         size=18)

# Escreve uma linha de texto no grafico; os primeiros dois parametros sao as
# coordenadas (x, y) [ou, neste caso, as coordenadas (r, theta)] para o
# inicio da primeira letra do texto, o terceiro parametro eh a string de
# texto em si, e o quarto parametro controla o tamanho da fonte.
ax_polar2.text(-1.1*np.pi/2, 0.7, 'Magnitude', size=18)

# Move o titulo ligeiramente para cima, para que nao se sobreponha ao
# indicador 'theta = 0'
titulo_polar2.set_y(1.09)

# plt.savefig('PadraoRad2_edit.png')

# %% Plotando os coeficientes de Fourier dos padroes de radiação

# Usa uma serie de Fourier de 3ª ordem
N_fourier = 3
k = np.arange(-N_fourier, N_fourier+1)

# Usa a funcao para obter os coeficientes de Fourier complexos
ck_1 = AnaliseFourier(padrao_rad1, N_fourier)
ck_2 = AnaliseFourier(padrao_rad2, N_fourier)

# Plota a amplitude e fase dos coeficientes
plt.figure()
plt.subplot(211)
plt.stem(k, np.abs(ck_1))
plt.ylabel('Amplitude')
plt.title('Serie de Fourier - Padrao Rad 1')
plt.ylim([-0.1, 0.6])
plt.xlim([-3.5, 3.5])
plt.subplot(212)
plt.stem(k, np.angle(ck_1))
plt.ylabel('Fase [rad]')
plt.xlabel('Indice do Coeficiente')
plt.ylim([-np.pi*1.1, np.pi*1.1])
plt.xlim([-3.5, 3.5])
# plt.savefig('CoefFourier1.png')

plt.figure()
plt.subplot(211)
plt.stem(k, np.abs(ck_2))
plt.ylabel('Amplitude')
plt.title('Serie de Fourier - Padrao Rad 2')
plt.ylim([-0.1, 0.4])
plt.xlim([-3.5, 3.5])
plt.subplot(212)
plt.stem(k, np.angle(ck_2))
plt.ylabel('Fase [rad]')
plt.xlabel('Indice do Coeficiente')
plt.ylim([-np.pi*1.1, np.pi*1.1])
plt.xlim([-3.5, 3.5])
# plt.savefig('CoefFourier2.png')

# %% Ressintetizar os padroes de radiacao com os coeficientes calculados

# Calcula os padroes de radiacao a partir dos coeficientes de Fourier obtidos

rad_ressintese1 = SinteseFourier(ck_1, phi.shape[0])
rad_ressintese2 = SinteseFourier(ck_2, phi.shape[0])

# Plota o padrao de radiação original e o ressintetizado
plt.figure()
ax_123 = plt.subplot(111, polar=True)
plt.plot(phi, np.abs(padrao_rad1), label='Original')
plt.plot(phi, np.abs(rad_ressintese1), 'r--', label='Ressintese')
plt.title('Ressintese do Padrao de Radiacao 1')
plt.legend(loc='lower left')
ax_123.set_rlim([0, 2.5])


plt.figure()
ax_456 = plt.subplot(111, polar=True)
plt.plot(phi, np.abs(padrao_rad2), label='Original')
plt.plot(phi, np.abs(rad_ressintese2), 'r--', label='Ressintese')
plt.title('Ressintese do Padrao de Radiacao 2')
plt.legend(loc='lower left')
ax_456.set_rlim([0, 1.])
