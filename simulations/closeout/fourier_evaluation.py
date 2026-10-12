"""Fast evaluation of the same finite Fourier polynomials.

Uniform FFT nodes and ANALYTIC first derivatives feed periodic cubic Hermite
interpolation. Its exact-arithmetic uniform error is <=h^4 sup|psi''''|/384.
At 65536 nodes this is below float64 roundoff for M<=2048. It changes neither
coefficients, normalization, nor Fourier truncation; canonical Clenshaw is
retained for the frozen M128 production stream and independent quadrature.
"""
from functools import lru_cache
import numpy as np
from simulations.dgp.rough import coefficients


@lru_cache(maxsize=3)
def table(M,n=65536):
    c=coefficients(M);frequency=np.arange(M+1)
    spectrum=np.zeros(n//2+1,complex)
    spectrum[:M+1]=n/2*c*(-1.)**frequency
    values=np.fft.irfft(spectrum,n)
    derivative=spectrum.copy();derivative[:M+1]*=1j*np.pi*frequency
    derivatives=np.fft.irfft(derivative,n)
    bound=(2/n)**4*np.sum(np.abs(c)*(np.pi*frequency)**4)/384
    if bound>1e-15:raise ValueError('Interpolation error is not below the declared floating tolerance.')
    return values,derivatives,float(bound)


def psi(u,M):
    values,derivatives,_=table(M);n=len(values);h=2/n
    coordinate=((np.asarray(u)+1)%2)/h
    left=np.floor(coordinate).astype(np.int64)%n;fraction=coordinate-np.floor(coordinate)
    right=(left+1)%n;t=fraction;t2=t*t
    return ((2*t-3)*t2+1)*values[left]+((t-2)*t+1)*t*h*derivatives[left]+(-2*t+3)*t2*values[right]+(t-1)*t2*h*derivatives[right]


def beta_fast(z,p,M):
    angle=np.pi*(z[:,0]+1)+p.eta*psi(z[:,1:]-z[:,:1],M).sum(axis=1)
    return np.sqrt(p.c_beta)*np.column_stack([np.sqrt(2/3)*np.cos(angle),np.sqrt(2/3)*np.sin(angle),np.full(len(z),1/np.sqrt(3))])
