import numpy as np


def Hamiltonian(mu, delta, k):
    J = 1 
    hz= J * np.cos(k) + mu/2
    hy= delta * np.sin(k)
    H=np.zeros((2,2), dtype=complex)
    
    sigma_x = np.matrix([[0, 1], [1, 0]])
    sigma_y = np.matrix([[0, -1j], [1j, 0]])
    sigma_z = np.matrix([[1, 0], [0, -1]])

    H= hz*sigma_z +hy*sigma_y
    
    return H

def ki(i,L):
    return 2*np.pi*i/L

def Ek(mu, delta, k):
    J = 1 
    hz= J * np.cos(k) + mu/2
    hy= delta * np.sin(k)
    return np.sqrt(hz**2 +hy**2)

def winding_number(n_k, mu, delta):
    wn = 0.0
    
    eigenvecs = np.zeros((n_k, 2, 2), dtype=complex)
    for i in range(n_k):
        _, eigenvecs[i] = np.linalg.eigh(Hamiltonian(mu, delta, ki(i,n_k)))
     
    d0 = np.dot(np.conj(eigenvecs[n_k-1, 0, :]), eigenvecs[0, 1, :])
    d1 = np.dot(np.conj(eigenvecs[n_k-1, 1, :]), eigenvecs[0, 0, :])
    arg = np.angle(d0*d1)
    wn += arg/(n_k*np.pi)
    for i in range(n_k-1):
        d0 = np.dot(np.conj(eigenvecs[i, 0, :]), eigenvecs[i+1, 1, :])
        d1 = np.dot(np.conj(eigenvecs[i, 1, :]), eigenvecs[i+1, 0, :])

        arg = np.angle(d0*d1)
        wn += arg/(n_k*np.pi)
        
    wn = np.sign(wn)*1 if np.abs(wn)>0.95 else 0 # I wrote this on my own!  
    if wn==0 and mu<0: wn+=2
    return wn

def ck(mu, delta, k):
    J=1
    hz= J * np.cos(k) + mu/2.
    en=Ek(mu, delta, k)
    return 0.5 + hz/(2.*en)

def fk(mu, delta, k):
    J=1
    hy= delta * np.sin(k)
    en=Ek(mu, delta, k)
    return hy/(2.*en)


