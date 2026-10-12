"""Optional compiled evaluation of the unchanged kernel and stock averaging."""
import ctypes
from pathlib import Path
import subprocess
import platform
import shutil
import numpy as np
from simulations.closeout.numerical import CACHE
from simulations.provenance import file_hash


class ManagedFeatures:
    def __init__(self,anchors,threads=4):
        self.anchors=np.ascontiguousarray(anchors,dtype=np.float64)
        self.threads=threads
        source=Path(__file__).with_suffix('.c')
        extension='.dylib' if platform.system()=='Darwin' else '.so'
        library=CACHE/f'managed_{file_hash(source)[:16]}{extension}'
        if not library.exists():
            flags=['-framework','Accelerate','-dynamiclib'] if platform.system()=='Darwin' else ['-shared','-fPIC']
            compiler=shutil.which('clang') or shutil.which('cc')
            subprocess.run([compiler,'-O3','-fno-fast-math','-ffp-contract=off','-fno-math-errno',*flags,'-pthread',str(source),'-o',str(library)],check=True)
        self.library=ctypes.CDLL(str(library));self.function=self.library.managed
        double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.function.argtypes=[ctypes.c_int]*4+[double,double,double,integer,double,ctypes.c_int]
        self.function.restype=ctypes.c_int
    def __call__(self,z,returns,cuts):
        z=np.ascontiguousarray(z,dtype=np.float64);r=np.ascontiguousarray(returns,dtype=np.float64)
        if r.ndim==1:r=r[:,None]
        cuts=np.ascontiguousarray(cuts,dtype=np.int32)
        if z.ndim!=2 or z.shape[1]!=6 or len(r)!=len(z) or not 1<=r.shape[1]<=3 or np.any(cuts<=0) or np.any(cuts>len(z)) or np.any(np.diff(cuts)<=0):raise ValueError('Invalid managed-feature inputs.')
        out=np.empty((len(cuts),r.shape[1],len(self.anchors)),dtype=np.float64)
        code=self.function(len(z),len(self.anchors),r.shape[1],len(cuts),z,self.anchors,r,cuts,out,self.threads)
        if code:raise RuntimeError('Native managed feature calculation failed.')
        return out
