import numpy as np 

def SaturationADJ(hsiimg, function, *arg):
    hsiimg_ = hsiimg.copy()
    # H,S,I
    hsiimg_[..., 1] = function(hsiimg[..., 1], *arg)
    return hsiimg_

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gamma_correction(x:np.ndarray ,A=1, gamma=0.5):

    return (A*x)**gamma