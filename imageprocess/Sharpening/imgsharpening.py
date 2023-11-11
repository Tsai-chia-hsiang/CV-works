import numpy as np
import cv2 
import os

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



class IMGSharpener():
    
    def __init__(self) -> None:
        
        self.__lp_mask = np.array(
            [[0,-1,0],
            [-1,4,-1],
            [0,-1,0]], dtype=np.float64
        )
        
        self.__strong_lp_mask = np.array(
            [[-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1]], dtype=np.float64
        )
  
    def Laplacian_Sharpening(self, img:np.ndarray, strong_mask=False, A=1.0)->np.ndarray:
        
        if strong_mask:
            m = self.__strong_lp_mask.copy()
        else:
            m = self.__lp_mask.copy()
    
        m[1][1] =  A + m[1][1]

        e = Conv2D(img=(img.astype(np.float64)), m=m)
        e =  np.clip(e, 0, 255).astype(np.uint8)
        return e
    
   
def makepath(p):
    if not os.path.exists(p):
        os.mkdir(p)
    return p


def read_test_images(root)->list:
    imginfo = []
    for r, _, f in os.walk(root):
        for fi in f:
            fs = os.path.join(r, fi)
            dpos = fi.rfind(".")
            
            imginfo.append(
                {
                    'fname':fi,
                    'name':fi[:dpos],
                    'img':cv2.imread(fs,cv2.IMREAD_GRAYSCALE)
                }
            )
        
    return imginfo

def main(testimgroot, resultroot):
    
    testimgs_info = read_test_images(testimgroot)
    Sharpener=IMGSharpener()
    
    A = [1.0 ,1.5, 1.7, 2.0]
    print(f"Using High boost filter : A = {A} ")
    for imginfo in testimgs_info:
        print()
        print(imginfo['fname'])
        
        
        savefolder = makepath(
            os.path.join(resultroot,f"{imginfo['name']}")
        )
        
        print("Mask center = -4 : ")
        elist = []
        for a in A:
            elist.append( 
                Sharpener.Laplacian_Sharpening(
                    img=imginfo['img'] , strong_mask=False, A=a
                )
            )
        
        
        s4 = makepath(os.path.join(savefolder,"center_neg_4"))
        print(f"save at : {s4}")
        for e,a in zip(elist,A):
            ret = cv2.imwrite(
                os.path.join(s4, f"A{a*10:.0f}{imginfo['fname']}"),e
            )
            if imginfo['fname'][-3:] != 'bmp':
                ret = cv2.imwrite(
                os.path.join(s4, f"A{a*10:.0f}{imginfo['fname'][:-3]}bmp"),e
            )
        
        
        print("Mask center = -8 : ")
        e1list = []
        for a in A:
            e1list.append( 
                Sharpener.Laplacian_Sharpening(
                    img=imginfo['img'], strong_mask=True, A=a
                )
            )
            
        s8 = makepath(os.path.join(savefolder,"center_neg_8"))
        print(f"save at : {s8}")
        for e1, a in zip(e1list, A):
            ret = cv2.imwrite(
                os.path.join(s8, f"A{a*10:.0f}{imginfo['fname']}"),e1
            ) 
            if imginfo['fname'][-3:] != 'bmp':
                ret = cv2.imwrite(
                os.path.join(s8, f"A{a*10:.0f}{imginfo['fname'][:-3]}bmp"), e1
            ) 
        
        

if __name__ == "__main__":
    
    testimgroot=os.path.join("HW2_test_image")
    resultroot = makepath(os.path.join("result"))
    
    
    main(testimgroot=testimgroot, resultroot=resultroot)
