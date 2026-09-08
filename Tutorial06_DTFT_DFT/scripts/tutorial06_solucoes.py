"""
Exemplo de solução para o Tutorial 06 de Processamento Digital de Sinais e Aplicações em Acústica

https://github.com/fchirono/Aulas_PDS_Acustica

Autor:
    Fabio Casagrande Hirono
    Fev 2026
"""

import numpy as np
from scipy.io import wavfile

import matplotlib.pyplot as plt
plt.close('all')


def DTFT(x, Omega):
    """ Realiza uma DTFT do sinal 'x' nas frequências normalizadas
    especificadas em 'Omega' """
    out = np.zeros(Omega.shape, 'complex')
    n = np.arange(x.shape[0])

    out = np.dot(x, np.exp(-1j*np.outer(n, Omega)))

    return out


def DTFT_alt(x, Omega):
    """ Realiza uma DTFT do sinal 'x' sobre o vetor de frequências
    normalizadas 'Omega' usando uma computação alternativa com loops 'for' """

    x_freq = np.zeros(Omega.shape[0], 'complex')

    N = np.arange(x.shape[0])

    for k in range(Omega.shape[0]):
        x_freq[k] = np.dot(x, np.exp(-1j*N*Omega[k]))

    return x_freq


def DFT(x, N_DFT):
    """ Realiza uma DFT do sinal 'x', com 'N_DFT' coeficientes """

    # Verifica se o sinal é mais longo ou mais curto que a DFT e ignora
    # as amostras extras ou preenche com zeros
    if x.shape[0] > N_DFT:
        x = x[:N_DFT]
    elif x.shape[0] < N_DFT:
        x = np.concatenate((x, np.zeros(N_DFT - x.shape[0])), axis=0)

    nk = np.arange(N_DFT)
    matrix = np.exp(-1j*2*np.pi/N_DFT*np.outer(nk, nk))
    
    out = np.dot(matrix, x[0:N_DFT])
    return out


def IDFT(X, N_DFT):
    """ Realiza uma IDFT do espectro 'X', com 'N_DFT' coeficientes """
    nk = np.arange(N_DFT)
    matrix = np.exp(1j*2*np.pi/N_DFT*np.outer(nk, nk))/N_DFT
    
    out = np.dot(matrix, X[0:N_DFT])
    return out


def IDFT2(X, N_DFT):
    """ Realiza uma IDFT do espectro 'X', com 'N_DFT' coeficientes, usando
    um loop for """
    nk = np.arange(N_DFT)
    out = np.zeros(N_DFT, 'complex')  # Muito importante declará-lo complexo!
    for n in range(N_DFT):
        out[n] = np.dot(np.exp(1j*2*np.pi/N_DFT*n*nk)/N_DFT, X)
    return out


# flag para salvar figuras
save_figs = False


# %% Parte 1 : DTFT

# leia um arquivo wav e propriedades adicionais
fs_wav, ai_sinal_int16 = wavfile.read('A1.wav')
_, a2_sinal_int16 = wavfile.read('A2.wav')

# normalização dos dados de áudio de 16 bits
N_bits = 16
a1 = ai_sinal_int16/(2.**(N_bits-1)-1)
a2 = a2_sinal_int16/(2.**(N_bits-1)-1)

plt.figure()
plt.plot(a1)
plt.title('Sinal a1')

plt.figure()
plt.plot(a2)
plt.title('Sinal a2')

# crie o vetor de frequência angular normalizada
Omega = np.linspace(-np.pi, np.pi, 4096)
#Omega = np.pi*np.arange(-1, 1+2./10000, 2./10000)

# implementação "DTFT_alt" é menos computacionalmente intensiva
X1 = DTFT_alt(a1, Omega)
X2 = DTFT_alt(a2, Omega)

# plote os dados
plt.figure(figsize=(8, 5))
#plt.plot(Omega, 20*np.log10(np.abs(X1)), label='A1')
#plt.plot(Omega, 20*np.log10(np.abs(X2)), label='A2')
plt.plot(Omega, np.abs(X1), label='A1')
plt.plot(Omega, np.abs(X2), label='A2')
plt.title('DTFTs')
plt.xlabel('Frequência Angular Normalizada [rad]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.tight_layout()
if save_figs:
    plt.savefig('DTFTA1A2.png')

# crie o vetor de frequência temporal
f = Omega*fs_wav/(2*np.pi)

# plote a DTFT de 'A1.wav' em Hz
plt.figure(figsize=(8, 5))
plt.plot(f, np.abs(X1), label='A1')
plt.title('DTFTs')
plt.xlabel('Frequência temporal [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.tight_layout()
if save_figs:
    plt.savefig('DTFTA1_Hz.png')


print("resolução de frequência:, {:1.6f} Hz / {:1.6f} rad/amostra".
      format(f[3]-f[2], Omega[3]-Omega[2]))

# use faixa de 3*pi para Omega
Omega3 = np.linspace(-3*np.pi, 3*np.pi, 4096)
#Omega3 = 3*np.pi*np.arange(-1, 1+2./10000, 2./10000)

X1_3pi = DTFT_alt(a1, Omega3)
plt.figure(figsize=(8, 5))
plt.plot(Omega3/np.pi, np.abs(X1_3pi))
plt.title('DTFT Periódica')
plt.xlabel('$\Omega/\pi$', size=15)
plt.ylabel('Magnitude')
plt.grid()
plt.tight_layout()
if save_figs:
    plt.savefig('DTFT_periodica.png')

# %% Parte 2 : DFT

# gerando sequência aleatória
L = 20
ruido = np.random.randn(L)

# Use tamanhos diferentes de DFT
DFT_Ruido1 = DFT(ruido, L)
DFT_Ruido2 = DFT(ruido, L-1)
DFT_Ruido3 = DFT(ruido, L+1)

plt.figure()
plt.stem(ruido)
plt.xlabel('n [amostras]')
plt.ylabel('Amplitude')
plt.xlim([-1, L+1])
plt.ylim([1.2*ruido.min(), 1.2*ruido.max()])
#plt.title('Sinal Aleatório Discreto')


plt.figure()
plt.subplot(211)
plt.stem(np.abs(DFT_Ruido1))
plt.title('DFT Ruído (comprimento "L")')
plt.ylabel('Magnitude')
plt.ylim([-2, 10])
plt.xlim([-1, 21])
plt.subplot(212)
plt.stem(np.angle(DFT_Ruido1))
plt.ylabel('Fase [rad]')
plt.xlabel('k [índice de frequência]')
plt.ylim([-4, 4])
plt.xlim([-1, 21])
plt.tight_layout()
if save_figs:
    plt.savefig('DFT_L.png')

plt.figure()
plt.subplot(211)
plt.stem(np.abs(DFT_Ruido2))
plt.title('DFT Ruído (comprimento "L-1")')
plt.ylabel('Magnitude')
plt.ylim([-2, 10])
plt.xlim([-1, 21])
plt.subplot(212)
plt.stem(np.angle(DFT_Ruido2))
plt.ylabel('Fase [rad]')
plt.xlabel('k [índice de frequência]')
plt.ylim([-4, 4])
plt.xlim([-1, 21])
plt.tight_layout()
if save_figs:
    plt.savefig('DFT_Lmenos1.png')

plt.figure()
plt.subplot(211)
plt.stem(np.abs(DFT_Ruido3))
plt.title('DFT Ruído (comprimento "L+1")')
plt.ylabel('Magnitude')
plt.ylim([-2, 10])
plt.xlim([-1, 21])
plt.subplot(212)
plt.stem(np.angle(DFT_Ruido3))
plt.ylabel('Fase [rad]')
plt.xlabel('k [índice de frequência]')
plt.ylim([-4, 4])
plt.xlim([-1, 21])
plt.tight_layout()
if save_figs:
    plt.savefig('DFT_Lmais1.png')


# %%
# Calcule a DFT usando np.fft.fft
# --> dá o mesmo resultado que 'DFT_Ruido1' (até a precisão numérica)
FFT_ruido = np.fft.fft(ruido, L)

plt.figure()
plt.subplot(211)
plt.stem(np.abs(FFT_ruido))
plt.title('DFT (com numpy.fft.fft)')
plt.ylabel('Magnitude')
plt.ylim([-2, 14])
plt.xlim([-1, 21])
plt.subplot(212)
plt.stem(np.angle(FFT_ruido))
plt.ylabel('Fase [rad]')
plt.xlabel('Frequência')
plt.ylim([-4, 4])
plt.xlim([-1, 21])
plt.tight_layout()
if save_figs:
    plt.savefig('FFT_L.png')

# Calcule as IDFTs (resultados têm partes imaginárias minúsculas!)
ruido1 = IDFT(DFT_Ruido1, L)
ruido2 = IDFT(DFT_Ruido2, L-1)
ruido3 = IDFT(DFT_Ruido3, L+1)

# Calcule a IDFT usando "IDFT2" - numericamente idêntica à IDFT
ruido1_2 = IDFT2(DFT_Ruido1, L)

# Plote o sinal de ruído original e os resultados da IDFT
plt.figure()
plt.subplot(411)
plt.plot(ruido, 'C0', marker='o')
plt.title('Sinal de ruído original')
plt.xlim([-1, L+1])
plt.grid()
plt.subplot(412)
plt.plot(np.real(ruido2), 'C1', marker='o')
plt.title('IDFT de comprimento "L-1"')
plt.xlim([-1, L+1])
plt.grid()
plt.subplot(413)
plt.plot(np.real(ruido1), 'C0', marker='o')
plt.title('IDFT de comprimento "L"')
plt.xlim([-1, L+1])
plt.grid()
plt.subplot(414)
plt.plot(np.real(ruido3), 'C2', marker='o')
plt.title('IDFT de comprimento "L+1"')
plt.xlim([-1, L+1])
plt.grid()
plt.tight_layout()
if save_figs:
    plt.savefig('IDFT.png')

# %% Parte 3.2

# Leia o sinal 'A1' entre as amostras 5100 e 6400 (freq. de amostragem é 'fs_wav')
d = a1[5099:6400]

# recrie o vetor Omega
Omega2 = np.linspace(-np.pi, np.pi, 4096)

# Calcule a DTFT
D_DTFT = DTFT_alt(d, Omega2)

# Plote o espectro sobre a frequência temporal (Hz)
plt.figure()
plt.plot(Omega2*(fs_wav/2)/np.pi, np.abs(D_DTFT), label='DTFT')
plt.title('DTFT e DFT do sinal A1 entre as amostras 5100 e 6400')
plt.ylabel('Magnitude')
plt.xlabel('Frequência [Hz]')

# calcule a DFT de 'd' usando o mesmo comprimento e rearranje o resultado para
# corresponder ao resultado da DTFT
N = d.shape[0]
D_DFT = DFT(d, N)
D_DFT_rearranged = np.concatenate((D_DFT[int(np.ceil(N/2.)):],
                                   D_DFT[0:int(np.floor(N/2.))+1]),
                                  axis=0)

# Compare o rearranjo com o 'np.fft.fftshift' integrado
# --> a diferença entre eles deve ser zero
D_DFT_fftshift = np.fft.fftshift(D_DFT)

# Gere o vetor de frequência aproximadamente simétrico (em Hz)
# --> note que N_DFT é par!
f_DFT = np.linspace(-fs_wav/2.+fs_wav/(2.*N), fs_wav/2. - fs_wav/(2.*N), N)

# Plote a DFT sobre a figura anterior
# --> os pontos stem devem cair exatamente sobre a linha da DTFT
plt.stem(f_DFT, np.abs(D_DFT_rearranged), linefmt='r', markerfmt='ro',
         label='DFT')
plt.legend()
plt.xlim([-200, 200])
plt.tight_layout()
if save_figs:
    plt.savefig('DTFT_DFT.png')
