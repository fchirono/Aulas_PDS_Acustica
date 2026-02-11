"""
Exemplo de solucoes para o Tutorial 03 de Processamento Digital de Sinais e Aplicações em Acústica

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2025
"""

import numpy as np

import matplotlib.pyplot as plt
plt.close("all")

import sounddevice as sd

save_flag = False

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
plt.legend(loc='lower right')
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.xlim([0, 0.1])
plt.ylim([-1.5, 1.5])
plt.title("Sinal discreto vs continuo")

if save_flag:
    plt.savefig("Ex1_visualizacao.png")

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


if save_flag:
    plt.savefig("Ex2_sinc.png")


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

if save_flag:
    plt.savefig("Ex3_reconstrucao.png")


# %% demonstracao de aliasing: sub-amostragem

plt.figure()

plt.subplot(311)
x_aliasing1 = np.cos(2*np.pi*250*t_c)
plt.plot(t_c, x_aliasing1)
plt.plot(t_d, x_aliasing1[::20], 'ro')
plt.xlim([0, 0.025])

# sd.play(x_aliasing1/10, fs_c, blocking=True)
# sd.play(x_aliasing1[::20]/10, fs_d, blocking=True)

plt.subplot(312)
x_aliasing2 = np.cos(2*np.pi*450*t_c)
plt.plot(t_c, x_aliasing2)
plt.plot(t_d, x_aliasing2[::20], 'ro')
plt.xlim([0, 0.025])

# sd.play(x_aliasing2/10, fs_c, blocking=True)
# sd.play(x_aliasing2[::20]/10, fs_d, blocking=True)

plt.subplot(313)
x_aliasing3 = np.cos(2*np.pi*750*t_c)
plt.plot(t_c, x_aliasing3)
plt.plot(t_d, x_aliasing3[::20], 'ro')
plt.plot(t_c, np.cos(2*np.pi*250*t_c), ':')
plt.xlim([0, 0.025])

# sd.play(x_aliasing3/10, fs_c, blocking=True)
# sd.play(x_aliasing3[::20]/10, fs_d, blocking=True)

if save_flag:
    plt.savefig("Ex4_subamostragem.png")

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
    plt.title(f"f1 = {f1} Hz / fs = 1000 Hz")
    
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

    if save_flag:
        plt.savefig(f"Ex5_aliasing_{f1}Hz.png")

