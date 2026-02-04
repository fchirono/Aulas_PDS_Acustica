# -*- coding: utf-8 -*-
"""
Funcoes e script para calcular a resposta ao impulso a um sistema contendo:
    - Uma fonte sonora tipo monopolo (omnidirecional);
    - Uma ou duas paredes refletoras;
    - e um receptor sonoro omnidirecional.
    
Atividades:

    1) Escreva duas funcoes para criar a resposta ao impulso de um sistema
    tipo atraso ideal:
        a) no dominio do tempo (atraso aproximado para numero inteiro de amostras);
        b) no dominio da frequencia (atraso exato).
    
    *Dica*: para a funcao no dominio da frequencia, use a funcao
    "numpy.fft.fftfreq" para criar o vetor de frequencias - essa funcao inclui
    frequencias positivas (entre 0 e fs/2) e negativas (entre -fs/2 e 0), ao
    inves de frequencias estritamente positivas (entre 0 e fs). O uso das
    frequencias estritamente positivas gera artefatos ao simular atrasos
    fracionais.
    
    
    2) Compare as RIs obtidas com a resposta analitica para um sistema atraso
    ideal, dada por:
        
        sinc(n-n0) = sin(pi*(n-n0)) / (pi*(n-n0)),
        
    onde 'n0' eh o atraso em amostras.
    
    
    
    3) Compute a resposta ao impulso e a resposta em frequencia de um sistema
    tipo fonte-parede-receptor com as seguintes dimensoes:
        
                 Parede
    ---------------------------------      ---
                                            | Dfp
        o))                 x              ---
        Fonte               Receptor
        
        |-------------------|
                Dfr
    
    
        a) Escreva a resposta ao impulso analitica do sistema
        b) Escreva a resposta em frequencia analitica do sistema
        c) Demonstre que a magnitude-ao-quadrado da resposta em frequencia
            varia de forma senoidal, e determine a "frequencia" desta oscilacao.
        
        (Ver solucao PDF em anexo na pasta)
        
        d) Mostre a resposta ao impulso e resposta em frequencia do sistema
        para Dfr = 2 m e Dfp = 1 m, e auralise a resposta deste sistema usando
        uma gravacao da sua propria voz como sinal de entrada.
        
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


# %% Exercicio 1

def cria_impulso_atrasado(T_atraso, T_duracao, fs, t_ou_f='f'):
    """
    Funcao para criar um sinal tipo impulso atrasado com amplitude unitaria e 
    atraso 'T_atraso' (em segundos), e frequencia de amostragem 'fs'. O calculo
    pode ser realizado no dominio do tempo ou da frequencia.
    
    
    Parametros
    ----------
    T_atraso : float
        Atraso do impulso, em segundos.
    
    T_duracao : float
        Duracao do sinal, em segundos.
    
    fs : float
        Frequencia de amostragem, em Hz.

    t_ou_f : {"t", "f"}
        Flag indicando se o calculo sera realizado no dominio do tempo ("t") 
        ou da frequencia ("f"). O valor padrao eh "f".

    Retorna
    -------
    t : (N,)-array
        Vetor de amostras no tempo, em segundos
        
    IR_atraso : (N,)-array
        Sinal tipo impulso atrasado, em amostras
    
    Notas
    -----
    Para calculo no dominio do tempo, a localizacao do impulso atrasado eh
    ajustada para a amostra mais proxima.
    
    Para calculos no dominio da frequencia, a localizacao do impulso atrasado
    eh exata.
    """
    
    assert t_ou_f in ["t", "f"], "Opcao invalida - 't_ou_f' deve ser 't' ou 'f'!"
    
    # intervalo de amostragem (segundos)
    dt = 1/fs
    
    # numero total de amostras no sinal - 2x a duracao desejada
    N_amostras  = int(T_duracao*fs)
    
    # vetor de amostras no tempo
    t = np.linspace(0, T_duracao-dt, N_amostras)
    
    if t_ou_f == "t":
        
        # encontre a amostra temporal mais proxima a "T_atraso"
        n_impulso = np.argmin( np.abs(t - T_atraso))
        
        # cria o sinal tipo impulso atrasado
        IR_atraso = np.zeros(N_amostras)
        IR_atraso[n_impulso] = 1
    
    elif t_ou_f == "f":

        # usa fftfreq e fftshift para calcular os atrasos
        
        # cria vetor de frequencias com 'N_amostras', contendo frequencias
        # positivas (ate fs/2) e negativas (apos fs/2)
        f = np.fft.fftfreq(N_amostras, dt)
        
        # Calcula um atraso na frequencia, e tira a Transformada Inversa para
        # retornar ao dominio do tempo
        Xf = np.exp(-1j * 2*np.pi * f * T_atraso)
        
        # # Usa a IFFT para calcular a Resposta ao Impulso do sinal atraso
        IR_atraso = np.fft.ifft(Xf, n=N_amostras)
        
    return t, IR_atraso.real



# %% Criar resposta ao impulso atrasada usando os dois metodos

# Frequencia de amostragem, em Hz
fs = 48000

# intervalo de amostragem no tempo, em segundos
dt = 1/fs

# T_atraso = 1253*dt         # atraso com numero inteiro de amostras
T_atraso = 1253.45*dt       # atraso com numero fracional de amostras

T_duracao = 0.1

t, xt = cria_impulso_atrasado(T_atraso, T_duracao, fs, 't')
t, xf = cria_impulso_atrasado(T_atraso, T_duracao, fs, 'f')

N_amostras = t.shape[0]


plt.figure()
plt.stem(t, xt, 'o-', label='Calc. tempo')
plt.stem(t, xf, 'x--', label='Calc. freq')
plt.grid()
plt.legend()
plt.ylabel("Amplitude")
plt.xlabel("Tempo [s]")

plt.title("RIs calculadas no tempo e na frequencia")

# %% Compara as RIs com a Resposta ao Impulso analitica ideal (funcao sinc
#   atrasada)

nt = np.arange(N_amostras)
n_atraso = T_atraso/dt

xf_analitico = np.sinc(nt - n_atraso)

nt2 = np.arange(100*N_amostras)/100
xf_analitico2 = np.sinc(nt2 - n_atraso)

plt.figure()
plt.plot(t, xf, 'o', color='C0', markersize=10, label='Freq')
plt.plot(t, xf_analitico, 's', color='C1', label='Freq. analitico')
plt.plot(nt2*dt, xf_analitico2, ':', color='C1', label="Freq interp")
plt.grid()
plt.legend()
plt.title("RIs calculada e analitica (sinc)")


# %% Exercicio 2d:

fs = 48000    # [Hz]

T_duracao = 0.1     # [s]

N_amostras_eco = int(fs*T_duracao)

c0 = 340    # [m/s]

# Configuracao 1: comb filtering
Dfr = 1     # [m]
Dfp = 0.5     # [m]

# # Configuracao 2: eco bem perceptivel
# Dfr = 3    # [m]
# Dfp = 3     # [m]

T_direto = Dfr/c0

D_eco = 2*np.sqrt( (Dfr/2)**2 + Dfp**2)
T_eco = D_eco/c0

t_eco, impulso_direto = cria_impulso_atrasado(T_direto, T_duracao, fs, 'f')
t_eco, impulso_eco = cria_impulso_atrasado(T_eco, T_duracao, fs, 'f')

RI_eco = impulso_direto/Dfr + impulso_eco/D_eco

plt.figure()
plt.plot(t_eco, RI_eco)
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.title("Resposta ao Impulso")


RF_eco = np.fft.rfft(RI_eco)
df_eco = fs/N_amostras_eco
f_eco = np.linspace(0, fs-df_eco, N_amostras_eco)[:N_amostras_eco//2+1]

plt.figure()
plt.plot(f_eco, np.abs(RF_eco)**2)
plt.grid()
plt.xlabel("Frequencia [Hz]")
plt.ylabel("Magnitude-ao-quadrado")
plt.title("Resposta em Frequencia")

#%% grava som pelo microfone do computador

# Use 'sd.query_devices()' para interrogar os canais de audio disponiveis, e escolha
# o canal disponivel no seu sistema para fazer a gravacao

# print("Gravando som no microfone...")
# y_mic = sd.rec(5*fs, fs, channels=1, device=1, blocking=True)
# print("Gravacao concluida!")

# sd.play(y_mic, fs, blocking=True)

# # convolucao entre sinal gravado e RI do sistema
# y_eco = np.convolve(y_mic[:, 0], RI_eco)

# plt.figure()
# plt.plot(y_mic)
# plt.plot(y_eco)

# sd.play(y_eco, fs, blocking=True)

# %% Avalie o efeito do sistema a um sinal tipo ruido branco

rng = np.random.default_rng()
ruidobranco = rng.normal(loc=0, scale=0.15, size=5*fs)

# sd.play(ruidobranco, fs, blocking=True)

# convolucao entre sinal gravado e RI do sistema
ruido_eco = np.convolve(ruidobranco, RI_eco)

plt.figure()
plt.plot(ruidobranco)
plt.plot(ruido_eco)

# sd.play(ruido_eco, fs, blocking=True)

