"""
Exercício exemplo para demonstrar o efeito de janelamento e 'zero-padding'

Referência
    K. Shin, J. Hammond
    "Fundamentals of Signal Processing for Sound and Vibration Engineers"
    John Wiley and Sons, 2008

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
import matplotlib.pyplot as plt


plt.rc('text', usetex=True)
plt.close('all')

# %% variáveis preliminares

A = 2           # amplitude
f0 = 1          # frequência fundamental
Tp = 1./f0      # período fundamental
fs = 10./Tp     # frequência de amostragem (10 amostras por Tp)
dt = 1./fs      # resolução temporal

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# escolha um dos 4 casos de exemplo:
#
# - Caso 1: senoides simples, comprimentos da DFT são múltiplos exatos dos períodos
#
# - Caso 2: senoides simples, comprimentos da DFT são múltiplos não-inteiros dos períodos
#
# - Caso 3: separando duas senoides, sem janelamento
#
# - Caso 4: separando duas senoides com janelamento
#
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

caso = 1

if caso == 1:
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # CASO 1 - comprimentos da DFT são números exatos de períodos
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    T1 = 1*Tp       # um período
    T2 = 5*Tp       # cinco períodos

    # sinal 1 (10 amostras)
    t1 = np.linspace(0, T1 - dt, int(T1*fs))
    N1 = np.size(t1)
    x1 = A*np.cos(2*np.pi*f0*t1)

    # sinal 2 (50 amostras)
    t2 = np.linspace(0, T2 - dt, int(T2*fs))
    N2 = np.size(t2)
    x2 = A*np.cos(2*np.pi*f0*t2)

elif caso == 2:
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # CASO 2 - comprimentos da DFT são números não-inteiros de períodos
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    T1 = 1.5*Tp     # 1,5 períodos
    T2 = 3.5*Tp     # 3,5 períodos

    # sinal 1 (15 amostras)
    t1 = np.linspace(0, T1 - dt, int(T1*fs))
    N1 = np.size(t1)
    x1 = A*np.cos(2*np.pi*f0*t1)

    # sinal 2 (35 amostras)
    t2 = np.linspace(0, T2 - dt, int(T2*fs))
    N2 = np.size(t2)
    x2 = A*np.cos(2*np.pi*f0*t2)

elif caso == 3:
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # CASO 3 - Separando ondas senoidais (sem janelas)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    T1 = 2.5*Tp     # 2,5 períodos
    T2 = 5.5*Tp     # 5,5 períodos

    f0b = f0*2*np.sqrt(2)
    Ab = A/20.              # ~ -26 dB abaixo de A

    # sinal 1 (25 amostras)
    t1 = np.linspace(0, T1 - dt, int(T1*fs))
    N1 = np.size(t1)
    x1 = A*np.cos(2*np.pi*f0*t1) + Ab*np.cos(2*np.pi*f0b*t1)

    # sinal 2 (55 amostras)
    t2 = np.linspace(0, T2 - dt, int(T2*fs))
    N2 = np.size(t2)
    x2 = A*np.cos(2*np.pi*f0*t2) + (A/10.)*np.cos(2*np.pi*f0b*t2)

elif caso == 4:
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # CASO 4 - Separando ondas senoidais usando janelas
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    T1 = 2.5*Tp     # 2,5 períodos
    T2 = 5.5*Tp     # 5,5 períodos

    f0b = f0*2*np.sqrt(2)
    Ab = A/20.              # ~ -26 dB abaixo de A

    # sinal 1 (25 amostras)
    t1 = np.linspace(0, T1 - dt, int(T1*fs))
    N1 = np.size(t1)
    win1 = np.hanning(N1)
    x1 = (A*np.cos(2*np.pi*f0*t1) + Ab*np.cos(2*np.pi*f0b*t1))*win1

    # sinal 2 (55 amostras)
    t2 = np.linspace(0, T2 - dt, int(T2*fs))
    N2 = np.size(t2)
    win2 = np.hanning(N2)
    x2 = (A*np.cos(2*np.pi*f0*t2) + (A/10.)*np.cos(2*np.pi*f0b*t2))*win2

# %% plotar os sinais no domínio do tempo

plt.figure()
for n in range(-2, 3):
    # plotar repetições periódicas
    plt.plot(t1 + n*T1, x1, 'o-', color='0.75')

    # unir repetições
    plt.plot([t1[-1] + n*T1, (n+1)*T1],
             [x1[-1], x1[0]], '--', color='0.75')

# plotar período fundamental
plt.plot(t1, x1, 'bo-')

# plotar resposta da janela
if caso == 4:
    plt.plot(t1, A*win1, 'g--')

plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('x1')
plt.ylim(-3, 3)


plt.figure()
for n in range(-2, 3):
    # plotar repetições periódicas
    plt.plot(t2 + n*T2, x2, 'o-', color='0.75')

    # unir repetições
    plt.plot([t2[-1] + n*T2, (n+1)*T2],
             [x2[-1], x2[0]], '--', color='0.75')

# plotar período fundamental
plt.plot(t2, x2, 'bo-')

# plotar resposta da janela
if caso == 4:
    plt.plot(t2, A*win2, 'g--')

plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('x2')
plt.ylim(-5, 5)

# %% preenchimento com zeros ('zero-padding') dos sinais no domínio do tempo
# --> usar 5000 pontos (aproximação da DTFT)

N_zp = 5000
T_zp = N_zp*dt
t_zp = np.linspace(0, T_zp-dt, N_zp)

x1_zp = np.concatenate((x1, np.zeros(N_zp-N1)))
x2_zp = np.concatenate((x2, np.zeros(N_zp-N2)))

# plotar os sinais
plt.figure()
for n in range(-2, 3):

    # plotar repetição periódica
    plt.plot(t_zp + n*T_zp, x1_zp, 'o-', color='0.75')

    # unir gráficos
    plt.plot([t_zp[-1] + n*T_zp, (n+1)*T_zp],
             [x1_zp[-1], x1_zp[0]], '--', color='0.75')
plt.plot(t_zp, x1_zp, 'ro-')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('x1 (com zero-padding)')
plt.ylim(-3, 3)

plt.figure()
for n in range(-2, 3):

    # plotar repetição periódica
    plt.plot(t_zp + n*T_zp, x2_zp, 'o-', color='0.75')

    # unir gráficos
    plt.plot([t_zp[-1] + n*T_zp, (n+1)*T_zp],
             [x2_zp[-1], x2_zp[0]], '--', color='0.75')
plt.plot(t_zp, x2_zp, 'ro-')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('x2 (com zero-padding)')
plt.ylim(-5, 5)

# %% calcular as DFTs

# sinal 1
df1 = fs/N1
freq1 = np.linspace(0, fs-df1, N1)
X1 = np.fft.fft(x1, N1)

# sinal 2
df2 = fs/N2
freq2 = np.linspace(0, fs-df2, N2)
X2 = np.fft.fft(x2, N2)

# sinais com zero-padding
df_zp = fs/N_zp
freq_zp = np.linspace(0, fs-df_zp, N_zp)
X1_zp = np.fft.fft(x1_zp)
X2_zp = np.fft.fft(x2_zp)


# plotar as DFTs
plt.figure()
plt.plot(freq1, np.abs(X1)/N1, 'bo', label='DFT (N={})'.format(N1))
plt.plot(freq_zp, np.abs(X1_zp)/N1, 'r--', label='DFT (N={})'.format(N_zp))
plt.ylabel('Magnitude Escalonada')
plt.xlabel('Freq [Hz]')
plt.title('|X1|')
plt.legend()

plt.figure()
plt.plot(freq2, np.abs(X2)/N2, 'bo', label='DFT (N={})'.format(N2))
plt.plot(freq_zp, np.abs(X2_zp)/N2, 'r--', label='DFT (N={})'.format(N_zp))
plt.ylabel('Magnitude Escalonada')
plt.xlabel('Freq [Hz]')
plt.title('|X2|')
plt.legend()


# plotar em dB
plt.figure()
plt.plot(freq1, 20*np.log10(np.abs(X1)/N1), 'bo',
         label='DFT (N={})'.format(N1))
plt.plot(freq_zp, 20*np.log10(np.abs(X1_zp)/N1), 'r--',
         label='DFT (N={})'.format(N_zp))
plt.ylabel('Magnitude Escalonada [dB]')
plt.xlabel('Freq [Hz]')
plt.title('|X1| (em dB)')
plt.ylim(-60, 20)
plt.legend()

plt.figure()
plt.plot(freq2, 20*np.log10(np.abs(X2)/N2), 'bo',
         label='DFT (N={})'.format(N2))
plt.plot(freq_zp, 20*np.log10(np.abs(X2_zp)/N2), 'r--',
         label='DFT (N={})'.format(N_zp))
plt.ylabel('Magnitude Escalonada [dB]')
plt.xlabel('Freq [Hz]')
plt.title('|X2| (em dB)')
plt.ylim(-60, 20)
plt.legend()