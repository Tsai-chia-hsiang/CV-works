import numpy as np

def Conv2D(img:np.ndarray, m:np.ndarray)->np.ndarray:
            
    def padding(img:np.ndarray, extnum=(1,1))->np.ndarray:
        padding_img = np.copy(img)
        padding_aix = [np.vstack, np.hstack]
        for i, m in enumerate(padding_aix):
            s = padding_img.shape[1-i]
            p = np.zeros((extnum[0],s))
            if i%2 == 1:
                p = p.T
            padding_img = m((p, padding_img, p))
        return padding_img

    mask = np.flip(m, (0,1))
    pimg = padding(img=img.astype(np.float64), extnum=(mask.shape[0]//2,mask.shape[1]//2))
    s = pimg.shape
    block_row_indices = np.arange(s[0]-(mask.shape[0]//2)*2).reshape(-1,1) + np.arange(mask.shape[0])
    block_col_indices = np.arange(s[1]-(mask.shape[1]//2)*2).reshape(-1,1) + np.arange(mask.shape[1]) 
  
    block_row = pimg[block_row_indices,:]
    blocks = np.transpose(block_row, (0,2,1))[:, block_col_indices]
    blocks = np.transpose(blocks, (0,1,3,2))
    return ((blocks*mask).sum(axis=(2,3)))



class Sharpener():
    
    def __init__(self) -> None:
        
        self.__lp_mask = np.array(
            [[0,-1,0],
            [-1,4,-1],
            [0,-1,0]], 
            dtype=np.float64
        )
        
        self.__strong_lp_mask = np.array(
            [[-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1]], 
            dtype=np.float64
        )
  
    def Laplacian_Sharpening(self, img:np.ndarray, strong_mask=False, A=1.0)->np.ndarray:
        m=None
        if strong_mask:
            m = self.__strong_lp_mask.copy()
        else:
            m = self.__lp_mask.copy()
    
        m[1][1] =  A + m[1][1]
        e = Conv2D(img=(img.astype(np.float64)), m=m)
        e =  np.clip(e, 0, 255).astype(np.uint8)
        return e


class RGBSharpener(Sharpener):
    
    def __init__(self) -> None:
        super().__init__()
    
    def Laplacian_Sharpening(self, img: np.ndarray, strong_mask=False, A=1, **kwarg) -> np.ndarray:
        
        img_ = np.dstack(
            [
                super().Laplacian_Sharpening(img[...,0], strong_mask, A),
                super().Laplacian_Sharpening(img[...,1], strong_mask, A),
                super().Laplacian_Sharpening(img[...,2], strong_mask, A)
            ]
        )
        
        return img_
