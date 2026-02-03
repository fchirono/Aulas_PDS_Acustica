"""
Exercicio demonstrando amostragem temporal e aliasing

Referencia:
    K. Shin, J. Hammond
    "Fundamentals of Signal Processing for Sound and Vibration Engineers"
    John Wiley and Sons, 2008

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np

rng = np.random.default_rng()

import scipy.signal as ss
import matplotlib.pyplot as plt
plt.close("all")

import sounddevice as sd

# %% plotar um sinal deterministico em tempo discreto e tempo "continuo"

# freq de amostragem 1 (tempo discreto)
fs_d = 1000
dt_d = 1/fs_d

# freq de amostragem (aproximacao de tempo "continuo")
fs_c = 20*fs_d
dt_c = 1/fs_c

# duracao do sinal [s]
T = 1.0

# freq do sinal [Hz]
f0 = 53

# base de tempo
t_d = np.linspace(0, T-dt_d, int(T*fs_d))
t_c = np.linspace(0, T-dt_c, int(T*fs_c))

x_d = np.cos(2*np.pi*f0*t_d)
x_c = np.cos(2*np.pi*f0*t_c)


plt.figure()
plt.plot(t_d, x_d, 'ro', label='Tempo discreto')
plt.plot(t_c, x_c, linestyle=':', color=[0.65, 0.65, 0.65],
         label='Tempo continuo')
plt.grid()
plt.legend()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.xlim([0, 0.1])
plt.title("Sinal discreto vs continuo")

# %% demonstrar que interpolacao tipo sinc de uma unica amostra possui valor
# zero nos instantes das outras amostras
# --> np.sinc(x) = sen(pi*x)/(pi*x)

# criar sinal contendo apenas 15a amostra
x_d_15 = np.zeros(x_d.shape)
x_d_15[15] = x_d[15]

# interpolar a 15a amostra no tempo "continuo"
x_c_interp15 = x_d[15]*np.sinc((t_c - t_d[15])*fs_d)

plt.figure()
plt.plot(t_d, x_d_15, 'ro', label="Tempo discreto")
plt.plot(t_c, x_c_interp15, linestyle=':', color=[0.65, 0.65, 0.65],
         label="Tempo continuo")
plt.title("Interpolando uma unica amostra")
plt.legend()
plt.grid()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.xlim([0, 0.03])

"""
Note que a funcao sinc corresponde a um sinal discreto com uma unica amostra
nao-zero, e portanto pode ser usado para reconstruir sinais continuos a partir
de amostras em tempo discreto.

Porem, note tambem que a funcao sinc em tempo continuo NAO eh causal, e assume
valores nao-zero antes de t=0 - ou seja, amostras no presente sofrem influencia
de amostras no futuro. Como sistemas reais devem necessariamente ser causais,
observa-se que a interpolacao pode ser apenas aproximada em sistemas fisicos.
"""

# %% reconstrucao ideal usando funcoes sinc (Teorema de Shannon - Sec. 5.5)

x_reconstruido = np.zeros(x_d.shape)

plt.figure()
plt.plot(t_d[:15], x_d[:15], 'ro', label="Sinal discreto")

for ni in range(15):
    x_temp = x_d[ni]*np.sinc((t_c - t_d[ni])*fs_d)
    plt.plot(t_c, x_temp,
             linestyle=':', color=[0.65, 0.65, 0.65])
    
    x_reconstruido += x_temp[::20]
    
plt.plot(t_d, x_reconstruido, linestyle='--', color='C1',
         label="Sinal continuo (reconstruido)")

plt.legend()
plt.grid()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.title("Reconstrucao de sinal digital")
plt.xlim([-0.005, 0.02])

"""
As diversas funcoes sinc geram valor nao-zero apenas no seu centro, e zeros na
localizacao das outras amostras. Desta forma, as amostras nao interferem entre
si nos instantes de amostragem.

Porem, note que as funcoes sinc decaem de forma bem lenta, de forma que mesmo
uma unica amostra resulta em uma funcao definida de t=-inf a +inf!
"""

# %% visualizacao de aliasing

for f1 in [150, 350, 550, 750]:
    
    x_aliasing = np.cos(2*np.pi*f1*t_d)
    
    f = np.fft.fftfreq(x_d.shape[0], d=1/fs_d)
    X_aliasing = np.fft.fft(x_aliasing, norm='forward')
    
    plt.figure()
    plt.subplot(211)
    plt.plot(t_d, x_aliasing, 'o')
    plt.plot(t_c, np.cos(2*np.pi*f1*t_c), 'k--')    
    
    if f1 > fs_d/2:
        plt.plot(t_c, np.cos(2*np.pi*(f1-fs_d)*t_c), ':')
    
    plt.xlim([0, 0.025])
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Frequencia: {f1} Hz")
    
    plt.subplot(212)
    plt.plot(f, np.abs(X_aliasing))
    plt.plot(f + fs_d, np.abs(X_aliasing),
             linestyle='--', color=[0.65, 0.65, 0.65])
    plt.plot(f - fs_d, np.abs(X_aliasing),
             linestyle='--', color=[0.65, 0.65, 0.65])
    
    plt.vlines([-f1, +f1], 0, 1, colors='k', linestyles='--')
    plt.xlabel("Frequencia [Hz]")
    plt.ylabel("Magnitude")    
    plt.tight_layout()


# %% demonstracao de aliasing: amostragem do tipo "cada N amostras" ()

plt.figure()

plt.subplot(311)
x_aliasing2 = np.cos(2*np.pi*250*t_c)
plt.plot(t_c, x_aliasing2)
plt.plot(t_d, x_aliasing2[::20], 'ro')
plt.xlim([0, 0.025])

plt.subplot(312)
x_aliasing2 = np.cos(2*np.pi*450*t_c)
plt.plot(t_c, x_aliasing2)
plt.plot(t_d, x_aliasing2[::20], 'ro')
plt.xlim([0, 0.025])

plt.subplot(313)
x_aliasing2 = np.cos(2*np.pi*750*t_c)
plt.plot(t_c, x_aliasing2)
plt.plot(t_d, x_aliasing2[::20], 'ro')
plt.plot(t_c, np.cos(2*np.pi*250*t_c), ':')
plt.xlim([0, 0.025])


# %% exemplo de aliasing: serie de Fourier

A_aliasing = [1, 0.6, 0.4]
f_aliasing = [150, 300, 600]


x_d_ex3 = np.zeros(x_d.shape)
x_c_ex3 = np.zeros(x_c.shape)

for n in range(3):
    x_d_ex3 += A_aliasing[n]*np.cos(2*np.pi*f_aliasing[n]*t_d)
    x_c_ex3 += A_aliasing[n]*np.cos(2*np.pi*f_aliasing[n]*t_c)
    

# sd.play(x_c_ex3/10, fs_c, blocking=True)

# sd.play(x_d_ex3/10, fs_d, blocking=True)


