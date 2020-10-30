
"""
Llibreria Python APAV
Guillem Martinez-illescas Ruiz
05/2020

"""

# imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs
import soundfile as sf
import sounddevice as sd
import struct

########################################################

# funció de lectura d'un fitxer WAV

########################################################

def lectura_WAV(fitxer):
    
    # fm: freqüència de mostratge
    
    f = open(fitxer, 'rb') # Obrim el fitxer en mode de lectura i binari
    
    formato = '<24si' # Eliminem els primers 24 bytes i agafem els 4 bytes següents (és la fm)
    
    dades = f.read(struct.calcsize(formato)) # Llegim la fm
    
    cc, fm = struct.unpack(formato, dades) # Traduïm la fm
    
    formato = '12si'
    
    dades = f.read(struct.calcsize(formato))
    
    cc, num_bytes = struct.unpack(formato, dades) # Total de mostres
    
    num_mostres = num_bytes//2 # Cada mostra són 2 bytes. Per tant, el nombre de mostres seran el total de bytes entre dos
    
    num_mostres = str(num_mostres) # Passem a string per fer el format
    
    formato = '<' + num_mostres + 'h' # Construïm el format
    
    dades = f.read(struct.calcsize(formato)) # Calculem les dades
    
    mostres = struct.unpack(formato, dades) # Traduïm les mostres a valors enters
    
    f.close() # Tanquem el fitxer 
    
    return fm, mostres # retornem la fm i les mostres d'audio


########################################################
    
# funció de lectura d'un fitxer WAV
    
########################################################


def escriptura_WAV(fitxer, freq_mostratge, senyal_audio):
    
    f = open(fitxer, 'wb') # Obrim el fitxer a escriure
    
    num_mostres = len(senyal_audio) # Nombre de mostres del senyal d'audio
    
    bytes_senyal = num_mostres * 2 # número de bytes a codificar del senyal
    
    formato = 'i'
    
    bs_cod = struct.pack(formato, bytes_senyal)
    
    formato = str(num_mostres) + 'h' # Format per codificar l'àudio
    
    senyal_codificat = struct.pack(formato, *senyal_audio) # Senyal codificat 
    
    formato = 'i' # Codifiquem un enter de 4 bytes
    
    fm_cod = struct.pack(formato, freq_mostratge)
    
    cap1 = b'RIFFZ\xee\x02\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'
    cap2 = b'\x88X\x01\x00\x02\x00\x10\x00data'
    
    # Escrivim en el fitxer de sortida
    
    f.write(cap1)
    f.write(fm_cod)
    f.write(cap2)
    f.write(bs_cod)
    f.write(senyal_codificat)
    
    f.close() # Tanquem el fitxer
    
    
########################################################
    
# Funció de transformada de Fourier    
    
########################################################
    
    
def transformadaFourier(x,NF):
    # calcular NF valors de la TF entre 0-0.5
    if NF%2:
        N= 2*NF -1
    else:
        N=(NF-1)*2
    
    lx= len(x)
    
    if lx > N:
        lx=N
        print(f'El senyal de {lx} mostres es trunca a {N} mostres')
        
    
    #inicialitzem el senyal
    X= np.zeros(NF) +0j 
    
    # calculem la TF
    for k in range(NF):
        for n in range(lx):
            ec= np.exp(-2j*np.pi*k*n/N)
            X[k]= X[k] + x[n]*ec
    
    return X,N


########################################################
    
# Funció de transformada de Fourier en una unica frequencia   
    
########################################################


def TransformadaFourier_F1(x,F):
    
    lx= len(x)
    n = np.arange(lx)
      
    
    #inicialitzem el senyal
    X= 0 +0j 
    # calculem la TF
    ec= np.exp(-2j*np.pi*F*n)
    X= x @ ec


    
    return X


########################################################
    
# Clase filtre IIR   
    
########################################################


class FiltreIIR:
    


    def __init__(self, b, a, v=[]): # Creació del filtre
        self.b = b # Coeficients
        self.a = a
        self.L = len(b)
        self.v = v # Estat intern
        if len(v) != len(b): self.v = np.zeros(len(b)-1)
    
    def reset(self): # Inicialització a 0 de l’estat intern
        self.v = np.zeros(self.L -1)
    
    def __call__(self, x): # funció de filtratge
        if isinstance(x, (int, float, complex)): x = np.array([x])
        Lx = len(x)
        M = len(self.v)
        y = np.zeros(Lx)
        for n in range(Lx):
            y[n] = self.b[0] * x[n] + self.v[0] 
            for i in range(1,M):
                self.v[i-1] = self.b[i] * x[n] - self.a[i] * y[n] + self.v[i]  
            self.v[M-1] = self.b[M] * x[n] - self.a[M] * y[n]
        return y
    

    def __repr__(self):
        return f"FiltreFIR({self.b} \n {self.v})"
    
    
    def __str__(self):
        return f"Filtre FIR d'ordre M={self.L-1} \n L = {self.L} coeficients \n nom_filtre=FiltreFIR(b,v) \n nom_filtre.reset() per inicialitzar l'estat intern \n y = nom_filtre(x) per filtrar x"



########################################################
    
# Clase filtre FIR   
    
########################################################
        
    

class FiltreFIR:

    
    
    def __init__(self, b, v=[]): # Creació del filtre
        self.b = b # Coeficients
        self.L = len(b)
        self.v = v # Estat intern
        if len(v) != len(b): self.v = np.zeros(len(b)-1)
        
    def reset(self): # Inicialització a 0 de l’estat intern
        self.v = np.zeros(self.L -1)
        
    def __call__(self, x): # funció de filtratge
        if isinstance(x, (int, float, complex)): x = np.array([x])
        Lx = len(x)
        M = len(self.v)
        y = np.zeros(Lx)
        for n in range(Lx):
            y[n] = self.b[0] * x[n] + self.b[1:] @ self.v
            self.v[1:M] = self.v[0:M-1] #v[1:] = v[: -1]
            self.v[0] = x[n]
        return y
    
    def __repr__(self):
        return f"FiltreFIR({self.b} \n {self.v})"
    
    def __str__(self):
        return f"Filtre FIR d'ordre M={self.L-1} \n L = {self.L} coeficients\n nom_filtre=FiltreFIR(b,v)\n nom_filtre.reset() per inicialitzar l'estat intern \n y = nom_filtre(x) per filtrar x"





########################################################
    
# Funció per fer la plantilla del modul del filtre
    
########################################################
        
    
def plantilla_modul(Fp,ap,Fa,aa):
    # ajustament de valors
    dp= (10**(ap/20)-1)/(10**(ap/20)+1)
    da= 10**(-aa/20)
    
    plt.figure("plantilla del modul del filtre")

    plt.plot([0,Fp],[1+dp,1+dp],'r')
    plt.plot([0,Fp,Fp],[1-dp,1-dp,0],'r')
    plt.plot([Fa,Fa,0.5],[1,da,da],'r')
    plt.axis([0,0.5,0,1 + 2*dp])
    
    
########################################################
    
# Funció per fer la plantilla del guany del filtre
    
########################################################
    
    
    
def plantilla_guany(Fp,ap,Fa,aa):
    
    #ajustament de valors
    
    dp = (10**(ap/20)-1)/(10**(ap/20)+1)
    Ga = -aa
    Gp1 = 20 * np.log10(1+dp)
    Gp2 = 20 * np.log10(1-dp)
    
    plt.figure("plantilla del guany del filtre")
    
    plt.plot([0,Fp],[Gp1,Gp1],'r')
    plt.plot([0,Fp,Fp],[Gp2,Gp2,Ga],'r')
    plt.plot([Fa,Fa,0.5],[0,Ga,Ga],'r')
    plt.axis([0, 0.5, 1.5 * Ga, Gp1 + 1])   

    
    
    
    
    
      
########################################################
    
# Funció per veure els zeros i els pols
    
########################################################  
    
    
    
    
    
    
def zeros_pols(b, a = 1):
    z = np.roots(b)
    while np.abs(z).max() > 10:
        z = np.delete(z,np.abs(z).argmax())
        print('Hi ha un zero de modul mes gran a 10')
    if isinstance(a,(int,float,complex)): a = np.array([a])
    if len(a) == 1:
        p = np.zeros(len(b) + 0j)
    else:
        p = np.roots(a)
    F= np.arange(720)/720
    cir_uni = np.exp(2j * np.pi * F)
    plt.plot(np.real(cir_uni), np.imag(cir_uni), ':b')
    plt.plot(np.real(z),np.imag(z),'og', mfc='w')
    plt.plot(np.real(p),np.imag(p),'xr')
    
    plt.title('Diagrama zeros-pols')
    plt.ylabel('part imaginària')
    plt.xlabel('part real')
    plt.axis('square')
    plt.grid()
    return z, p
    
    
########################################################
    
# Funció per retornar un filtre FIR optim
    
########################################################
    
    
    

def fir_optim(Fp,ap,Fa,aa):
    
    da = 10**(-aa/20)
    dp = (10**(ap/20)-1)/(10**(ap/20)+1)
    
    Fideal = np.array([0,Fp,Fa,0.5])
    Hmideal = np.array([1,0])
    contrapes = np.array([da,dp])


    L = 25
    b = scs.remez(L, Fideal, Hmideal,contrapes) 
    H1 = np.abs(TransformadaFourier_F1(b,Fp))
    if H1 > 1-dp:
        while H1 > 1-dp:
            L -= 1
            b = scs.remez(L, Fideal, Hmideal,contrapes)
            H1 = np.abs(TransformadaFourier_F1(b,Fp))
    elif H1 < 1-dp:
        while H1 < 1-dp:
            L += 1
            b = scs.remez(L, Fideal, Hmideal,contrapes)
            H1 = np.abs(TransformadaFourier_F1(b,Fp))
    return b


