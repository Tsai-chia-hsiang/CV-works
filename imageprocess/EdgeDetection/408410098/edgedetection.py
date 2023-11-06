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


def Conv2D(img:np.ndarray, mask:np.ndarray)->np.ndarray:
        
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
   
    pimg = padding(img=img, extnum=(mask.shape[0]//2,mask.shape[1]//2))
    
    s = pimg.shape
    block_row_indices = np.arange(s[0]-2).reshape(-1,1) + np.arange(mask.shape[0])
    block_col_indices = np.arange(s[1]-2).reshape(-1,1) + np.arange(mask.shape[1]) 
    block_row = pimg[block_row_indices,:]
    blocks = np.transpose(block_row, (0,2,1))[:, block_col_indices]
    blocks = np.transpose(blocks, (0,1,3,2))
    return ((blocks*mask).sum(axis=(2,3)))


def RGB2GraySacle(cimg:np.ndarray)->np.ndarray:
    R = cimg[..., 0].astype(np.float64)
    G = cimg[..., 1].astype(np.float64)
    B = cimg[..., 2].astype(np.float64)
    Gray = np.clip(0, 255, (0.3*R+0.59*G+0.11*B))
    return Gray


edgex = np.array(
    [[1,0,-1],
    [2,0,-2],
    [1,0,-1]
    ], dtype=np.float64
)
edgey = edgex.T

def Sobel(img:np.ndarray)->np.ndarray:
    Gx = Conv2D(img=img, mask=edgex)
    Gy = Conv2D(img=img, mask=edgey)

    E = (Gx**2+Gy**2)**0.5
    return np.clip(0,255, E).astype(np.uint8)

def main():
    imgpathes = walkdir(os.path.join("HW4_test_image"))
    edgedetection_dir = makepath(os.path.join("edge_detection1"))

    for imgpath in imgpathes:
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        edgedetect = Sobel(RGB2GraySacle(cimg=img).astype(np.uint8))
        gf = os.path.join(edgedetection_dir, imgpath[imgpath.rfind("\\")+1:])
        cv2.imwrite(gf, edgedetect)

if __name__ == "__main__":
    main()