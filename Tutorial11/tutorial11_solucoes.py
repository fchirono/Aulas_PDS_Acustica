# -*- coding: utf-8 -*-
"""
Exercicio demonstrando a diferenca entre metodos de analise espectral:   
    
    - Periodograma: a DFT normalizada de um unico segmento de uma serie
    temporal. O periodograma **NAO** reduz a variancia da estimativa conforme
    a quantidade de dados aumenta;
    
    - Metodo de Welch: modificacao do metodo do periodograma, onde a serie
    temporal e subdividida em diversos segmentos com sobreposicao e janelamento,
    e a media das DFTs de cada segmento resulta no estimador de Welch. O metodo
    de Welch reduz a variancia da estimativa conforme a quantidade de
    dados aumenta;
    
Compara tambem as duas normalizacoes possiveis para o metodo de Welch:
    
    - **Espectro** de potencia: resulta em estimativas consistentes para a
    amplitude de sinais deterministicos (senoides), independente do tamanho da
    DFT;
    
    - **Densidade espectral** de potencia: resulta em estimativas consistentes
    para a densidade espectral de potencia de sinais nao-deterministicos
    (ruido), independente do tamanho da DFT.


Referencias:
    - K. Shin, J. Hammond, "Fundamentals of Signal Processing for Sound and
    Vibration Engineers" - Wiley
    
    - G. Heinzel et al, "Spectrum and spectral density estimation by the
    Discrete Fourier transform (DFT), including a comprehensive list of window
    functions and some new flat-top windows", 2002.
    https://holometer.fnal.gov/GH_FFT.pdf
   
   
https://github.com/fchirono/AulasDSP

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np

import scipy.signal as ss

import matplotlib.pyplot as plt
plt.close('all')


# %% define as funcoes

def periodograma(x, fs=1, nome_janela='boxcar', normalizacao='espectro'):
    
    Nx = x.shape[0]
    
    # vetor de frequencias (unilateral)
    df = fs/Nx
    f = np.linspace(0, fs-df, Nx)[:Nx//2+1]
    
    # ----------------------------------------------------------------
    janela = ss.get_window(nome_janela, Nx)
    
    # normalizacoes da funcao de janelamento
    S1 = np.sum(janela)
    S2 = np.sum(janela**2)

    # ----------------------------------------------------------------
    # DFT unilateral
    
    X_dft = np.fft.fft(janela * x)[:Nx//2+1]
    
    if normalizacao == 'espectro':
        return f, 2 * np.abs(X_dft)**2 / (S1**2)
    
    elif normalizacao == 'densidade':
        return f, 2 * np.abs(X_dft)**2 / (fs * S2)


def welch(x, Ndft=512, N_sobreposicao=256, nome_janela='boxcar',
          fs=1, normalizacao='espectro'):
    
    Nx = x.shape[0]
    
    # vetor de frequencias (unilateral)
    df = fs/Ndft
    f = np.linspace(0, fs-df, Ndft)[:Ndft//2+1]
    
    # ----------------------------------------------------------------
    janela = ss.get_window(nome_janela, Ndft)
    
    # normalizacoes da funcao de janelamento
    S1 = np.sum(janela)
    S2 = np.sum(janela**2)

    # ----------------------------------------------------------------
    # Calcular numero de "janelas" a serem calculadas, adicionar zeros ao sinal
    # caso necessario

    # tamanho do "passo"
    N_passo = Ndft - N_sobreposicao
    
    # numero de "janelas" contidas na duracao do sinal
    N_janelas = int(np.ceil((Nx - N_sobreposicao) / (Ndft - N_sobreposicao)))
    
    # numero de amostras contidas em N_janelas de Ndft amostras e N_sobreposicao
    N_total = N_janelas*N_passo + N_sobreposicao

    # adicionar zeros ("zero-pad") ao final do sinal, caso necessario
    if N_total > Nx:
        x = np.concatenate((x, np.zeros(N_total - Nx)))
    
    # ----------------------------------------------------------------    
    # Realizar o calculo da sobreposicao-e-soma
    
    X_espectro = np.zeros(Ndft//2+1)
    
    for n in range(N_janelas):
        x_segmento = x[n*N_passo : n*N_passo + Ndft]
        
        X_dft = np.fft.fft(janela * x_segmento)[:Ndft//2+1]
        
        X_espectro += np.abs(X_dft)**2 / N_janelas
        
    if normalizacao == 'espectro':
        return f, 2*X_espectro/(S1**2)
    
    elif normalizacao == 'densidade':
        return f, 2*X_espectro/(fs*S2)



# %% Gerar um sinal do tipo ruido branco de banda limitada

# frequencia de amostragem [Hz]
fs = 48000

# intervalo de amostragem [s]
dt = 1/fs

# duracao do sinal [s]
T = 2.0

# vetor de amostras no tempo [s]
t = np.linspace(0, T-dt, int(T*fs))

# cria uma instancia do Gerador de Numeros Aleatorios do modulo Numpy
rng = np.random.default_rng()

# cria uma serie temporal de numeros aleatorios com distribuicao normal
# (i.e. gaussiana), com media 0.0, desvio padrao 1.0, e (T*fs) amostras.
x_ruidobranco = rng.normal(loc=0.0, scale=1.0, size=int(T*fs))

# Cria o filtro passa-faixas
fc_baixa = 50  # Hz
fc_alta = 2000  # Hz

ordem = 4
b, a = ss.butter(ordem, [fc_baixa, fc_alta], btype='band', fs=fs)

# Filta o sinal ruido branco para obter um sinal de banda limitada
x_passafaixa = ss.lfilter(b, a, x_ruidobranco)

# compara os sinais no tempo
plt.figure()
plt.plot(t[:250], x_ruidobranco[:250], ':',  label='Ruido Branco')
plt.plot(t[:250], x_passafaixa[:250], '--', label='Ruido passa-faixa')
plt.grid()
plt.legend()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")


# %% calcular a DEP dos sinais usando duracoes diferentes com o periodograma

N_per = 512

f1, Sxx_1 = periodograma(x_passafaixa[:N_per], fs, normalizacao='densidade')
f2, Sxx_2 = periodograma(x_passafaixa[:2*N_per], fs, normalizacao='densidade')
f5, Sxx_5 = periodograma(x_passafaixa[:5*N_per], fs, normalizacao='densidade')
f10, Sxx_10 = periodograma(x_passafaixa[:10*N_per], fs, normalizacao='densidade')

plt.figure()
plt.semilogx(f10, 10*np.log10(Sxx_10), ':',
             label=f'N_per amostras (df={f1[1]:.1f} Hz)')
plt.semilogx(f5, 10*np.log10(Sxx_5), '-.',
             label=f'2*N_per  amostras (df={f2[1]:.1f} Hz)')
plt.semilogx(f2, 10*np.log10(Sxx_2), '--',
             label=f'5*N_per amostras (df={f5[1]:.1f} Hz)')
plt.semilogx(f1, 10*np.log10(Sxx_1),
             label=f'10*N_per amostras (df={f10[1]:.1f} Hz)')
plt.grid()
plt.legend()
plt.ylabel("Magnitude")
plt.title("Densidade Espectral de Potencia - Periodograma")
plt.xlim([10, fs/2])
plt.ylim([-100, -25])
plt.xlabel("Frequencia [Hz]")

"""
Conforme a duracao da serie temporal aumenta, o estimador tipo periodograma 
aumenta a resolucao em frequencia (distancia entre amostras adjacentes da DFT)
mas nao reduz a variancia da estimativa, e portanto nao converge para a
Densidade Espectral de Potencia teorica do sinal!
"""

# %% calcular a PSD dos sinais usando diferentes duracoes (mas mesma Ndft) com o metodo de Welch

Ndft = 512
N_sobreposicao = 256

f1w, Sxxw_1 = welch(x_passafaixa[:1000], Ndft, N_sobreposicao,
                       fs=fs, normalizacao='densidade')
f2w, Sxxw_2 = welch(x_passafaixa[:2000], Ndft, N_sobreposicao,
                       fs=fs, normalizacao='densidade')
f5w, Sxxw_5 = welch(x_passafaixa[:5000], Ndft, N_sobreposicao,
                       fs=fs, normalizacao='densidade')
f10w, Sxxw_10 = welch(x_passafaixa[:10000], Ndft, N_sobreposicao,
                       fs=fs, normalizacao='densidade')

plt.figure()
plt.semilogx(f1w, 10*np.log10(Sxxw_1), '-.',
             label=f'1000 amostras (df={f1w[1]:.1f} Hz)')
plt.semilogx(f2w, 10*np.log10(Sxxw_2), '-.',
             label=f'2000 amostras (df={f2w[1]:.1f} Hz)')
plt.semilogx(f5w, 10*np.log10(Sxxw_5), '--',
             label=f'5000 amostras (df={f5w[1]:.1f} Hz)')
plt.semilogx(f10w, 10*np.log10(Sxxw_10),
             label=f'10000 amostras (df={f10w[1]:.1f} Hz)')
plt.grid()
plt.legend()
plt.ylabel("Magnitude")
plt.title("Densidade Espectral de Potencia - Metodo de Welch")
plt.xlim([10, fs/2])
plt.ylim([-100, -25])
plt.xlabel("Frequencia [Hz]")

"""
Conforme a duracao da serie temporal aumenta, o estimador tipo Welch 
nao aumenta a resolucao em frequencia (distancia entre amostras adjacentes da
DFT) mas reduz a variancia da estimativa, e portanto converge para a
Densidade Espectral de Potencia teorica do sinal!

NOTA: neste caso em particular, note que o numero de pontos na DFT nao resulta em
uma resolucao em frequencia (aprox. 94 Hz) fina o bastante para "resolver" a
frequencia de corte inferior (50 Hz) do filtro passa-banda! Para melhor
observar este efeito, eh necessario aumentar o tamanho da DFT.
"""

# %% aplicar o metodo de Welch no sinal senoide + ruido passa-faixa, comparar o
# efeito de diferentes Ndfts e diferentes normalizacoes

# frequencia do sinal senoidal
f0 = 1234

# Amplitude RMS
Vrms = 2

# Amplitude de pico
Vpk = Vrms * np.sqrt(2)

x_senoidal = Vpk * np.sin(2 * np.pi * f0 * t)

# verificar amplitude RMS do sinal resultante:
Vrms_calculada = np.sqrt(np.mean(x_senoidal**2))
print(f"A amplitude RMS do sinal senoidal eh {Vrms_calculada:.2f}")

x_soma = x_senoidal + x_passafaixa


# calcular a PS e a PSD do sinal usando diferentes Ndfts

f1_soma, PSD_soma1 = welch(x_soma, Ndft, N_sobreposicao,
                           nome_janela='hann', fs=fs, normalizacao='densidade')
f2_soma, PSD_soma2 = welch(x_soma, 2*Ndft, 2*N_sobreposicao,
                           nome_janela='hann', fs=fs, normalizacao='densidade')
f5_soma, PSD_soma5 = welch(x_soma, 5*Ndft, 5*N_sobreposicao,
                           nome_janela='hann', fs=fs, normalizacao='densidade')
f10_soma, PSD_soma10 = welch(x_soma, 10*Ndft, 10*N_sobreposicao,
                             nome_janela='hann', fs=fs, normalizacao='densidade')

plt.figure()
plt.loglog(f1_soma, PSD_soma1,
             label = "Ndft amostras")
plt.loglog(f2_soma, PSD_soma2,
             label = "2*Ndft amostras")
plt.loglog(f5_soma, PSD_soma5,
             label = "5*Ndft amostras")
plt.loglog(f10_soma, PSD_soma10,
             label = "10*Ndft amostras")
plt.grid()
plt.legend()
plt.ylabel(r"Magnitude [$V^2$/Hz]")
plt.title("Densidade Espectral de Potencia - Metodo de Welch")
plt.xlim([10, fs/2])
plt.ylim([1e-8, 1e2])
plt.ylim()
plt.xlabel("Frequencia [Hz]")

"""
Note que a densidade espectral de potencia do sinal tipo ruido passa-faixa
se mantem constante, independente da escolha de Ndft. Por outro lado,
a magnitude do pico correspondente a potencia do sinal senoidal varia com a
escolha de Ndft. 

O valor de cada elemento da DEP corresponde a potencia contida dentro de uma
caixinha de largua df = fs/Ndft, dividido pela largura da caixinha. Assim, a
potencia do sinal senoidal pode ser obtida ao considerar a
integral (area) da DEP em uma faixa de frequencias ao redor do pico.

Uma primeira aproximacao pode ser simplesmente o produto da magnitude do pico
vezes a resolucao em frequencia df, correspondente a potencia contida dentro 
de uma unica caixinha da DEP centrada no pico.
"""

f1_soma, PS_soma1 = welch(x_soma, Ndft, N_sobreposicao,
                          nome_janela='hann', fs=fs, normalizacao='espectro')
f2_soma, PS_soma2 = welch(x_soma, 2*Ndft, 2*N_sobreposicao,
                          nome_janela='hann', fs=fs, normalizacao='espectro')
f5_soma, PS_soma5 = welch(x_soma, 5*Ndft, 5*N_sobreposicao,
                          nome_janela='hann', fs=fs, normalizacao='espectro')
f10_soma, PS_soma10 = welch(x_soma, 10*Ndft, 10*N_sobreposicao,
                            nome_janela='hann', fs=fs, normalizacao='espectro')


plt.figure()
plt.loglog(f1_soma, PS_soma1,
             label = "Ndft amostras")
plt.loglog(f2_soma, PS_soma2,
             label = "2*Ndft amostras")
plt.loglog(f5_soma, PS_soma5,
             label = "5*Ndft amostras")
plt.loglog(f10_soma, PS_soma10,
             label = "10*Ndft amostras")

# marcar a potencia do sinal senoidal com uma linha tracejada em Vrms^2
plt.hlines(Vrms**2, f1_soma[0], f1_soma[-1], colors='k', linestyles='--')

plt.grid()
plt.legend()
plt.ylabel(r"Magnitude [$V^2$]")
plt.title("Espectro de Potencia - Metodo de Welch")
plt.xlim([10, fs/2])
plt.ylim([1e-8, 1e2])
plt.ylim()
plt.xlabel("Frequencia [Hz]")

"""
O espectro de potencia, por outro lado, resulta em uma estimativa correta da 
potencia do sinal senoidal independentemente da escolha de Ndft, porem o
valor da potencia contida em um sinal de espectro continuo tipo ruido ira 
variar com a Ndft. 

Neste caso, o valor de cada elemento do EP corresponde a quantidade de energia
dentro de uma "caixinha" de largura igual a resolucao em frequencia df = 
fs/Ndft, de tal forma que resolucoes em frequencia mais grosseiras (caixinhas
largas) irao conter mais energia, enquanto resolucoes mais finas (caixinhas
finas) irao conter menos energia. 
                                                                  
Para obter a potencia total do sinal a partir do Espectro de Potencia, basta
somar os valores do Espectro de Potencia (nao eh necessario integrar - i.e.
levar em conta a largura da caixinha). 

NOTA:
    Pedir para os alunos explicarem por que as curvas de DEP/EP possuem
    diferentes graus de variancia com diferentes valores de Ndft!

    Resposta: a duracao total do sinal eh constante, entao Ndfts menores irao gerar
    um alto numero de janelas curtas, e portanto mais elementos para tirar a media
    do espectro. Por outro lado, Ndfts longas resultaram em menos janelas calculadas,
    e portanto menos elementos no calculo da media e um espectro mais erratico.
"""