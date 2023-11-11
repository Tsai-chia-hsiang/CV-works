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

sobelx = np.array(
    [[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]],
    dtype=np.float64
)
sobely = sobelx.T

def img_gradient(img:np.ndarray)->tuple:
    
    gx = np.clip(Conv2D(img, sobelx/8), 0, 255)
    gy = np.clip(Conv2D(img , sobely/8), 0, 255)
    
    return np.rad2deg(np.arctan2(gy, gx)), (gx**2+gy**2)**0.5

