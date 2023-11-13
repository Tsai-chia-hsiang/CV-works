import numpy as np
import cv2
from imageoperation import img_gradient, Conv2D

def color_histogram(img:np.ndarray)->np.ndarray:
    return np.vstack(
        [
            np.bincount(img[..., i].flatten(), minlength = 256) 
            for i in range(img.shape[2])
        ]
    )

def HoG(img_:np.ndarray)->np.ndarray:

    def interpolation_count(theta:np.ndarray, mag:np.ndarray)->np.ndarray:
        """
        Linear Interpolation:
        
        |_(k - bin_base-k)_|_(next_bin_base-k)_|
        
        k = ((next_bin_base-k)*bin_base + (k - bin_base-k)*next_base_base)
                /(next_bin_base-bin_base)
        """
        
        xbin = np.floor(theta/20).astype(np.int32)
        bins = np.zeros((9,))
        pre = (((xbin + 1)*20-theta)/20)*mag
        post = (((theta - xbin*20))/20)*mag
        for i in range(9):
            mask = (xbin == i)
            bins[i] += (pre[mask].sum())
            bins[(i+1)%9] += (post[mask].sum())

        return bins
    
    def generate_cells(theta:np.ndarray, mag:np.ndarray, w:int=8)->np.ndarray:

        cells = np.zeros((theta.shape[0]//w, theta.shape[1]//w , 9))

        for i in range(theta.shape[0]//w):
            for j in range(theta.shape[1]//w):
                cells[i,j] = interpolation_count(
                    theta=theta[i*8:(i+1)*8,j*8:(j+1)*8], 
                    mag=mag[i*8:(i+1)*8, j*8:(j+1)*8]
                )
            
        return cells

    def generate_blocks(cells:np.ndarray, w:int=2)->np.ndarray:
        
        blocks = np.zeros((cells.shape[0] - w//2, cells.shape[1] - w//2, cells.shape[2]*w*w))
        
        for i in range(cells.shape[0] - w//2):
            for j in range(cells.shape[1] - w//2):
                blocks[i,j] = cells[i:i+2,j:j+2].flatten()
                blocks[i,j] /= (np.sum(blocks[i,j]**2))**0.5
        
        return blocks

    img = img_ if img_.ndim == 2 else cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (64,128))

    theta, mag = img_gradient(img=img)
    shiftangle_index = np.where(theta >= 180.0)
    theta[shiftangle_index[0], shiftangle_index[1]] -= 180
    cells = generate_cells(theta=theta, mag=mag)
    
    return generate_blocks(cells=cells)

class Gabor_Filter():

    def __init__(self, ksize:tuple=(11,11), sigma:float=1.5, Nangle:int=6, gamma:float=1, omega:float=2*np.pi/3) -> None:
    
        self.kernels = [ 
            np.flip( 
                Gabor_Filter.generate_gabor_filter(
                    size=ksize,sigma=sigma, 
                    omega=omega, theta=t,
                    gamma=gamma
                ),(0,1)
            ) 
            for t in np.arange(0,np.pi, np.pi/Nangle)
        ]
        
    @staticmethod
    def generate_gabor_filter(size,sigma, omega, theta, gamma=1)->np.ndarray:
        center = (size[0]//2, size[1]//2)
        x,y = np.meshgrid(np.arange(0, size[0]),np.arange(0, size[1]))
        x = x - center[0]
        y = y - center[1]

        xp = x*np.cos(theta)+y*np.sin(theta)
        yp = y*np.cos(theta)-x*np.sin(theta)
        f = np.exp(-(xp**2 + yp**2 * gamma**2)/(2*sigma**2))
        real = np.cos(omega*(xp))
        K = (f*real)
        K /= np.sum(K)
        return K

    def filt(self, img:np.ndarray)->np.ndarray:

        I = np.zeros(img.shape)
        for k in self.kernels:
            #I = np.maximum(I, Conv2D(img, k))
            I += Conv2D(img, k)
        return I/I.max()*255
