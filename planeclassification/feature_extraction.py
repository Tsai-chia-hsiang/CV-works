import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, root :os.PathLike, shuffle:bool=False) -> None:
        
        self.classes, self.__datapathes = Dataset.walk_data_root(root=root)

    @staticmethod
    def walk_data_root(root:os.PathLike)->tuple:
        
        def walkdir(root:os.PathLike)->list:
            ret = []
            for r,_,fs in os.walk(root):
                for f in fs:
                    ret.append(osp.join(r, f))
            return ret
        
        c = [c.name for c in os.scandir(root)]
        ret = []
        for ciid, ci in enumerate(c):
            class_dir = osp.join(root, ci)
            x = walkdir(root= class_dir)
            ret += [(ciid, xi) for xi in x]
        return c, ret

    def __getitem__(self, i):
        return self.__datapathes[i]
    def __len__(self):
        return len(self.__datapathes)

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

def Gabor_filter(size,sigma, omega, theta)->np.ndarray:
    center = (size[0]//2, size[1]//2)
    x,y = np.meshgrid(np.arange(0, size[0]),np.arange(0, size[1]))
    x = x - center[0]
    y = y -center[1]
    N = (2*np.pi*sigma[0]*sigma[1])
    xp = x*np.cos(theta)+y*np.sin(theta)
    #yp = y*np.cos(theta)-x*np.sin(theta)
    f = np.exp(-((x/sigma[0])**2 + (y/sigma[1])**2)/2)
    real = np.cos(omega*(xp)+0)
    K = (f*real)/N
    return K

gabor_kernel = np.flip(
    Gabor_filter(
        size=(15,15),sigma=(5,5), omega=2*np.pi/10, 
        theta=np.pi/4
    ),
    (0,1)
)


def makedir(p:os.PathLike)->os.PathLike:
    if not osp.exists(p):
        os.mkdir(p)
    return p

def main():
    
    train_data = Dataset(root=osp.join("data", "train"))
    train_feature_dir = makedir(osp.join(makedir("features"), "train"))
    colorhis_dir = makedir(osp.join(train_feature_dir, "colorhistogram"))
    hog_dir = makedir(osp.join(train_feature_dir, "HoG"))
    gabor_dir =  makedir(osp.join(train_feature_dir, "Gabor"))

    for label, imgpath in tqdm(train_data):

        name = osp.split(imgpath)[-1]
        
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        gimg = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2GRAY)

        d_color_hist = makedir(osp.join(colorhis_dir,train_data.classes[label]))
        np.save(osp.join(d_color_hist,name[:-4]), color_histogram(img=img))

        d_hog = makedir(osp.join(hog_dir,train_data.classes[label]))
        np.save(osp.join(d_hog,name[:-4]),HoG(gimg))

        d_gabor = makedir(osp.join(gabor_dir,train_data.classes[label]))
        gaborimg = np.clip(Conv2D(gimg, gabor_kernel)*255, 0, 255).astype(np.uint8)
        cv2.imwrite(osp.join(d_gabor ,name), gaborimg)


if __name__ == "__main__":
    main()