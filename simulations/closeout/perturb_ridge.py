"""Reuse a response-one ridge eigensystem for tiny Fourier design perturbations.

The equation, penalties, normalization and solution are unchanged. Perturbed
normal equations are solved by a preconditioned fixed point for the coefficient
DIFFERENCE; this avoids cancelling entire population-sized normal equations.
Every result is checked using the actual perturbed design and the original
residual tolerance. No jitter, sign choice or new statistical estimator.
"""
import numpy as np
from scipy.linalg import eigh
from simulations.closeout.metrics import normal_residual


class RidgeFactor:
    def __init__(self,x):
        self.x=np.asarray(x,float);self.T,self.P=self.x.shape
        self.dual=self.T<self.P
        matrix=self.x@self.x.T/self.T if self.dual else self.x.T@self.x/self.T
        values,self.vectors=eigh(matrix)
        if values[0]<-max(values[-1],1e-30)*1e-12:raise ValueError('Empirical operator is not PSD.')
        self.values=np.maximum(values,0)
        self.projection=self.vectors.T@(np.ones(self.T) if self.dual else self.x.mean(axis=0))
    def path(self,penalties):
        lam=np.asarray(penalties,float)
        if np.any(lam<=0):raise ValueError('Positive penalties required.')
        a=self.vectors@(self.projection[:,None]/(self.values[:,None]+lam))
        if self.dual:a=self.x.T@a/self.T
        C=np.sum(self.values[:,None]/(self.values[:,None]+lam),axis=0)
        return a,C
    def inverse(self,r,lam):
        if not self.dual:return self.vectors@((self.vectors.T@r)/(self.values[:,None]+lam))
        inner=self.vectors.T@(self.x@r)
        return r/lam-self.x.T@(self.vectors@(inner/(self.values[:,None]+lam)))/(self.T*lam)
    def perturb(self,delta,penalties):
        D=np.asarray(delta,float);lam=np.asarray(penalties,float)
        base,_=self.path(lam);change=np.zeros_like(base)
        dm=D.mean(axis=0)[:,None]
        def dsecond(w):
            dx=D@w;xx=self.x@w
            return (self.x.T@dx+D.T@xx+D.T@dx)/self.T
        right=(D.T@(np.ones((self.T,1))-self.x@base-D@base)-self.x.T@(D@base))/self.T
        for iteration in range(30):
            corrected=np.column_stack([self.inverse((right-dsecond(change))[:,j:j+1],value)[:,0] for j,value in enumerate(lam)])
            difference=np.linalg.norm(corrected-change)
            change=corrected
            if difference<=1e-13*max(np.linalg.norm(base),1e-20):break
        else:raise ValueError('Perturbed ridge difference failed to converge; no tolerance relaxation.')
        result=base+change
        absolute,relative=normal_residual(self.x+D,result,lam)
        if absolute>1e-8 or relative>1e-8:raise ValueError('Perturbed normal equations exceed unchanged tolerances.')
        return result,dict(iterations=iteration+1,absolute_residual=absolute,relative_residual=relative,
                           relative_coefficient_change=float(np.linalg.norm(change)/np.linalg.norm(base)))
