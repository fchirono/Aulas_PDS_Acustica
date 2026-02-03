"""
Exemplo de solucoes para o Tutorial 2 de Processamento Digital de Sinais e Aplicações em Acústica

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2025
"""

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile, loadmat
import matplotlib.pyplot as plt

import tutorial2_funcoes as Tutorial2

plt.close('all')  # fecha todas as figuras abertas

# define se devemos salvar as figuras e arquivos WAV
salvar_figuras = True
salvar_wav = True


# %% Tutorial 1.1 - 1.3 - Gerando e plotando sinais

fs = 44100          # Frequencia de amostragem [Hz]
dt = 1./fs     # Periodo de amostragem [s]
T_max = 1.          # Duracao total do sinal []

# cria um array Numpy com as amostras do tempo [s]
t = np.linspace(0, T_max-dt, int(T_max/dt))

A = 1.                              # amplitude
freq = 200.                         # frequencia [Hz]
onda_senoidal = A*np.sin(2*np.pi*freq*t)   # cria onda senoidal

plt.figure()                    # open a new figure using pyplot
plt.plot(t, onda_senoidal)      # plota onda senoidal vs tempo
plt.xlabel('Tempo [s]')         # adiciona rotulo no eixo 'x'
plt.ylabel('Amplitude')         # adiciona rotulo no eixo 'y'
plt.xlim([0, t[500]])           # ajusta limites [min, max] do eixo x
plt.ylim([-A*1.1, A*1.1])       # ajusta limites [min, max] do eixo y

# adiciona um titulo para a figura contendo a frequencia da onda
plt.title("Onda senoidal, frequencia f = {:.2f} Hz".format(freq))

if salvar_figuras:
    plt.savefig('onda_senoidal.png')

# A funcao 'np.random.randn' gera amostras de ruido branco Gaussiano com media
# zero e desvio padrao unitario. O exemplo abaixo mostra como gerar amostras
# com media e desvio padrao arbitrarios:
rudio_media = 0.
ruido_desviopadrao = 1.
ruido_branco = ruido_desviopadrao*np.random.randn(t.shape[0]) + rudio_media

# Abre uma nova figura e plota as primeiras 500 amostras de cada sinal
plt.figure()
plt.plot(t[:500], ruido_branco[:500])
plt.xlim([0, t[500]])
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title('Ruido branco Gaussiano')
if salvar_figuras:
    plt.savefig('ruido_branco.png')

# %% Tutorial 1.3 - Operacoes com sinais

# soma dos dois sinais
soma_sinais = onda_senoidal + ruido_branco

plt.figure()
plt.plot(t[:500], soma_sinais[:500])
plt.xlim([0, t[500]])
plt.ylabel('Amplitude')
plt.xlabel('Tempo [s]')
plt.title('Soma dos dois sinais')
if salvar_figuras:
    plt.savefig('sinais_soma.png')

# multiplicacao (elemento a elemento) dos dois sinais
sinal_multiplicacao = onda_senoidal*ruido_branco

plt.figure()
plt.plot(t[:500], sinal_multiplicacao[:500])
plt.xlim([0, t[500]])
plt.ylabel('Amplitude')
plt.xlabel('Tempo [s]')
plt.title('Multiplicacao dos dois sinais')
if salvar_figuras:
    plt.savefig('sinais_mult.png')

# raiz quadrada do modulo (valor absoluto) dos sinais
seno_raiz = np.sqrt(np.abs(onda_senoidal))
ruido_raiz = np.sqrt(np.abs(ruido_branco))

# plota os dois novos sinais usando diferentes estilos de linha e marcadores,
# e adiciona um rotulo em cada um para criar a legenda
plt.figure()
plt.plot(t[:500], seno_raiz[:500], 'b--', label='Raiz seno',
         linewidth=3)
plt.plot(t[:500], ruido_raiz[:500], 'g-', label='Raiz ruido')
plt.xlim([0, t[500]])
plt.ylabel('Amplitude')
plt.xlabel('Tempo [s]')
plt.title('Raiz quadrada de cada sinal')
plt.legend()
if salvar_figuras:
    plt.savefig('sinais_raiz.png')


# quadrado (elemento a elemento) de cada sinal
seno_quadrado = onda_senoidal**2
ruido_quadrado = ruido_branco**2

plt.figure()
plt.plot(t[:500], seno_quadrado[:500], linestyle='--',
         label='Seno ao quadrado', linewidth=3)
plt.plot(t[:500], ruido_quadrado[:500], linestyle='-',
         label='Ruido ao quadrado')
plt.xlim([0, t[500]])
plt.ylabel('Amplitude')
plt.xlabel('Tempo [s]')
plt.title('Quadrado de cada sinal')
plt.legend()
if salvar_figuras:
    plt.savefig('sinais_ao_quadrado.png')

# %% Tutorial 2.1 - Sistemas

# Criar um impulso de amplitude unitaria na amostra n = 0
pulso_unitario1 = np.zeros(100)
pulso_unitario1[0] = 1.

# Criar um impulso de amplitude unitaria com atraso de 50 amostras
pulso_unitario2 = np.zeros(100)
pulso_unitario2[50] = 1

# Aplicar ambos os sinais de teste ao sistema sob análise
saida_tempo1 = Tutorial2.sistema1(pulso_unitario1)
saida_tempo2 = Tutorial2.sistema1(pulso_unitario2)

# plotar as entradas e saídas em função das amostras
plt.figure()
plt.subplot(211)
plt.stem(pulso_unitario1, markerfmt='bo', label='Entrada 1')
plt.stem(pulso_unitario2, linefmt='g--', markerfmt='gs',  label='Entrada 2')
plt.ylabel('Amplitude da Entrada')
plt.ylim([-0.5, 1.2])
plt.xlim([-1, 100])
plt.legend()
plt.title('Teste de (In)Variância Temporal')
plt.subplot(212)
plt.stem(saida_tempo1, label='Saída 1')
plt.stem(saida_tempo2, linefmt='g--', markerfmt='gs', label='Saída 2')
plt.ylim([-0.5, 1.2])
plt.xlim([-1, 100])
plt.xlabel('Amostras')
plt.ylabel('Amplitude da Saída')
plt.legend()

if salvar_figuras:
    plt.savefig('teste_tempo.png')

# Teste de linearidade: em um sistema linear, f(a)+f(b) = f(a+b)
saida_lin1 = Tutorial2.sistema2(onda_senoidal)
saida_lin2 = Tutorial2.sistema2(ruido_branco)
saida_lin12 = Tutorial2.sistema2(onda_senoidal + ruido_branco)

# Plotar ambos os sinais para comparação visual
plt.figure()
plt.plot(t[:500], saida_lin1[:500] + saida_lin2[:500], 'b', label='f(a)+f(b)')
plt.plot(t[:500], saida_lin12[:500], 'g--', label='f(a+b)')
plt.ylabel('Amplitude')
plt.xlabel('Tempo [s]')
plt.title('Teste de linearidade')
plt.xlim([0, t[500]])
plt.ylim([-4, 8])
plt.legend()

if salvar_figuras:
    plt.savefig('linearidade.png')


# %% Tutorial 3 - Energia e Potência de sinais

# carregar o conteúdo do arquivo MAT como um dicionário
arquivo_mat = loadmat('EnergiaPotencia')

# ler as chaves no dicionário para obter os arrays
sinal_x1 = arquivo_mat['x1']
sinal_x2 = arquivo_mat['x2']

# Calcular a energia e a potência do 1o sinal
energia_x1 = np.sum(sinal_x1**2)
potencia_x1 = energia_x1/sinal_x1.shape[0]

print("Energia x1: {:2f}".format(energia_x1))
print("Potencia x1: {:2f}".format(potencia_x1))

# calcular a potência do 2o sinal
# --> x2 é periódico, portanto tem energia infinita!
potencia_x2 = np.sum(sinal_x2**2)/sinal_x2.shape[0]
print("Potencia x2: {:2f}".format(potencia_x2))

# %% escrevendo arquivos wav

N_bits = 16	            	   # Número de bits
pico_wav = 2**(N_bits-1) - 1        # Amplitude de pico da amostra wav

# Alterar o tipo de dado para int de 16 bits e normalizar as amplitudes para 'pico_wav'
dados_wav_seno = np.int16(onda_senoidal*pico_wav)

# Escrever o arquivo 'wav'
if salvar_wav:
    wavfile.write('senoide.wav', fs, dados_wav_seno)
    
# normalizar a amplitude do ruído para 1
ruido_branco_normalizado = ruido_branco/np.abs(ruido_branco).max()

# Alterar o tipo de dado para int de 16 bits e normalizar as amplitudes para 'pico_wav'
dados_wav_ruido = np.int16(ruido_branco_normalizado*pico_wav)

# Escrever o arquivo 'wav'
if salvar_wav:
    wavfile.write('ruido_branco.wav', fs, dados_wav_ruido)