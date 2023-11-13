import os
import numpy as np
import cv2
from .kernel import gaussian_kernel, gabor_kernel
from .operation import Conv2D, img_gradient

class __Feature_Extractor():
    
    def __init__(self) -> None:
        pass
    
    def __call__(self,img:np.ndarray, need_return:True ,saveto:os.PathLike=None) -> np.ndarray|None:
        
        I=self.postprocess(self.extract(self.preprocess(img)))

        if saveto is not None:
            self.save(saveto=saveto, obj=I)
        
        if need_return:
            return I

    def extract(self, img:np.ndarray)->np.ndarray:
        raise NotImplementedError() 
    
    def preprocess(self, img:np.ndarray)->np.ndarray:
        return img
    
    def postprocess(self, img:np.ndarray)->np.ndarray:
        return img
    
    def to_gray(self, img:np.ndarray)->np.ndarray:
        match img.ndim:
            case 2:
                return img
            case 3:
                return cv2.cvtColor(img ,cv2.COLOR_RGB2GRAY)
            case _:
                raise ValueError('seens not an image ...')

    def save(self, saveto:os.PathLike, obj:np.ndarray)->None:
        np.save(saveto, obj)
    
class Color_Histogram(__Feature_Extractor):
    
    def __init__(self) -> None:
        super().__init__()
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        return np.vstack(
            [
                np.bincount(img[..., i].flatten(), minlength = 256) 
                for i in range(img.shape[2])
            ]
        )

class HoG(__Feature_Extractor):
    
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(self.to_gray(img=img), (64,128))

    def extract(self, img: np.ndarray) -> np.ndarray:

        theta, mag = img_gradient(img=img)
        
        shiftangle_index = np.where(theta >= 180.0)
        theta[shiftangle_index[0], shiftangle_index[1]] -= 180
        
        cells = self.__generate_cells(theta=theta, mag=mag)

        return self.__generate_blocks(cells)

    def __generate_cells(self, theta:np.ndarray, mag:np.ndarray, w:int=8)->np.ndarray:

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
        
        cells = np.zeros((theta.shape[0]//w, theta.shape[1]//w , 9))

        for i in range(theta.shape[0]//w):
            for j in range(theta.shape[1]//w):
                cells[i,j] = interpolation_count(
                    theta=theta[i*8:(i+1)*8,j*8:(j+1)*8], 
                    mag=mag[i*8:(i+1)*8, j*8:(j+1)*8]
                )
            
        return cells

    def __generate_blocks(self, cells:np.ndarray, w:int=2)->np.ndarray:
        
        blocks = np.zeros((cells.shape[0] - w//2, cells.shape[1] - w//2, cells.shape[2]*w*w))
        
        for i in range(cells.shape[0] - w//2):
            for j in range(cells.shape[1] - w//2):
                blocks[i,j] = cells[i:i+2,j:j+2].flatten()
                blocks[i,j] /= (np.sum(blocks[i,j]**2))**0.5+1e-10
        
        return blocks

class Gabor_Features_Extractor( __Feature_Extractor):

    def __init__(self, ksize:int=11, sigma:float=1.5, Nangle:int=6, gamma:float=1, omega:float=2*np.pi/3) -> None:
        super().__init__()
        self.kernels = gabor_kernel(
            ksize=ksize, parameters=[
                {'sigma':sigma, 'omega':omega, 'theta':t, 'gamma':gamma} 
                for t in np.arange(0,np.pi, np.pi/Nangle) 
            ] 
        ) 
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return self.to_gray(img=img)

    def extract(self, img: np.ndarray) -> np.ndarray:
        
        I = np.zeros(img.shape)
        for k in self.kernels:
            #I = np.maximum(I, Conv2D(img, k))
            I += Conv2D(img, k)
        return I/I.max()*255
        #return I
    
