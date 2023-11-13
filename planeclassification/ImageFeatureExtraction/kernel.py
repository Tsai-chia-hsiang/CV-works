from typing import Any
import numpy as np

sobelx = np.array(
    [[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]],
    dtype=np.float64
)
sobely = sobelx.T

def makegrid(ksize:int)->tuple:
    start = -(ksize-1)//2
    end = -start
    return np.meshgrid(np.arange(start, end+1, 1),np.arange(start, end+1, 1))

def gaussian_numerator(x:np.ndarray, y:np.ndarray, sigma:float)->np.ndarray:
    return np.exp(-(x**2+y**2)/(2 * (sigma**2)))

def normalize_kernel(m:np.ndarray)->np.ndarray:
    return m/np.sum(m)

def gaussian_kernel(ksize:int,sigma:float|list)->np.ndarray|list:
    
    x, y = makegrid(ksize=ksize)
    
    def generate(s:float)->np.ndarray:
        m = gaussian_numerator(x=x, y=y, sigma=s)/(2*np.pi*(sigma**2))
        return normalize_kernel(m)

    p = [sigma] if isinstance(sigma, float) else sigma
    ks = [generate(s=si) for si in p]
    
    return ks if len(ks)>1 else ks[0]

def gabor_kernel(ksize:int, parameters:dict|list)->np.ndarray|list:
    
    """
    sigma:float : standard deviation 
    
    omega:float : frequency
    
    theta:float : orientation
    
    gamma:float : ratio
    """
   
    x,y = makegrid(ksize=ksize)

    def generate(sigma:float, omega:float, theta:float, gamma:float)->np.ndarray:

        xp = x*np.cos(theta)+y*np.sin(theta)
        yp = y*np.cos(theta)-x*np.sin(theta)
        k = gaussian_numerator(x=xp, y=yp*gamma,sigma=sigma)*np.cos(omega*(xp))
        
        return normalize_kernel(k)
    
    p = [parameters] if isinstance(parameters, dict) else parameters
    ks = [
        generate(
            sigma=pi['sigma'], omega=pi['omega'],
            theta=pi['theta'], gamma= pi['gamma']
        ) for pi in p
    ]
    
    return ks if len(ks) > 1 else ks[0]    
    