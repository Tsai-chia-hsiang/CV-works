import os
import os.path as osp
from dataset import walkdir, Dataset
import numpy as np
import cv2
from tqdm import tqdm
from imageoperation import Conv2D, img_gradient

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


def extract(dset, saveroot:os.PathLike):
    
    colorhis_dir = makedir(osp.join(saveroot, "colorhistogram"))
    hog_dir = makedir(osp.join(saveroot, "HoG"))
    gabor_dir =  makedir(osp.join(saveroot, "Gabor"))

    for di in tqdm(dset):
        imgpath = di
        label = None
        class_name = ""
        if isinstance(di, tuple):
            label = di[0]
            imgpath = label
            class_name=dset._classes[label]
        
        name = osp.split(imgpath)[-1]
        d_color_hist = makedir(osp.join(colorhis_dir,class_name)) if len(class_name) else colorhis_dir
        d_gabor = makedir(osp.join(gabor_dir, class_name)) if len(class_name) else gabor_dir
        d_hog = makedir(osp.join(hog_dir, class_name)) if len(class_name) else hog_dir
        
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        gimg = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2GRAY)
        
        np.save(
            osp.join(d_color_hist,name[:-4]), 
            color_histogram(img=img)
        )
        np.save(
            osp.join(d_hog,name[:-4]),
            HoG(gimg)
        )
        cv2.imwrite(
            osp.join(d_gabor ,name), 
            np.clip(Conv2D(gimg, gabor_kernel)*255, 0, 255).astype(np.uint8)
        )

def main():
    
    train_data = Dataset(root=osp.join("data", "train"))
    train_feature_dir = makedir(osp.join(makedir("features"), "train"))
    extract(dset=train_data, saveroot=train_feature_dir)

    test_data = walkdir(root=osp.join("data","test"))
    test_feature_dir = makedir(osp.join(makedir("features2"), "test"))
    extract(dset=test_data, saveroot=test_feature_dir)

if __name__ == "__main__":
    main()