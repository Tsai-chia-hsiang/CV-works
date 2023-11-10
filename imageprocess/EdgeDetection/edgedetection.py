import numpy as np
import cv2
import os

def walkdir(root:os.PathLike):
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            files.append(os.path.join(r, f))
    return files

def makepath(p):
    if not os.path.exists(p):
        os.mkdir(p)
    return p


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
    pimg = padding(
        img=img.astype(np.float64), 
        extnum=(mask.shape[0]//2,mask.shape[1]//2)
    )
    s = pimg.shape
    block_row_indices = np.arange(s[0]-2).reshape(-1,1) + np.arange(mask.shape[0])
    block_col_indices = np.arange(s[1]-2).reshape(-1,1) + np.arange(mask.shape[1]) 
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


def RGBEdge(img:np.ndarray)->np.ndarray:
    
    gx = np.dstack(list(Conv2D(img[..., i], m=sobelx) for i in range(3)))
    gy = np.dstack(list(Conv2D(img[..., i], m=sobely) for i in range(3)))

    gxx = (gx**2).sum(axis = 2)
    gyy = (gy**2).sum(axis = 2)
    gxy = (gx*gy).sum(axis = 2)

    thetas=0.5*np.arctan(2*gxy/(gxx-gyy+1e-10))
    f = (0.5*((gxx+gyy) + np.cos(2*thetas)*(gxx-gyy) + 2*gxy*np.sin(2*thetas)))
    f[np.where(f < 0)] = 0
    return np.clip(f**0.5, 0, 255)

def main():
    imgpathes = walkdir(os.path.join("HW4_test_image"))
    edgedetection_dir = makepath(os.path.join("result1"))

    for imgpath in imgpathes:
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        edgedetect = RGBEdge(img=img).astype(np.uint8)
        gf = os.path.join(edgedetection_dir, imgpath[imgpath.rfind("\\")+1:])
        cv2.imwrite(gf, edgedetect)

if __name__ == "__main__":
    main()