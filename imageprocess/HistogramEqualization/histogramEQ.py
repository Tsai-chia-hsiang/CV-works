import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json 

class GrayLevel_histogramEqualizer :
    
    def __init__(self) -> None:
        pass
    
    def transform(self, img:np.ndarray, spatial="global", needhist=False, **kwarg)->np.ndarray:

        if spatial == "global":
            return self.__global_hiseq(
                img=img,need_hist=needhist
            )
        elif spatial == "local":
            return self.__local_hiseq(
                img = img,need_hist=needhist, 
                bsize=kwarg['bsize']
            )

    def __count(self, img:np.ndarray)->np.ndarray:
        """
        return p[rk] = nk
        """ 
        p = np.bincount(img.flatten(), minlength=256)
        return p
    
    def __hiseq(self,img:np.ndarray)->np.ndarray:

        p = self.__count(img=img)
        cdf= np.cumsum(p/img.size)
        hist = (np.clip(cdf*255, 0, 255)).astype(np.uint8)
        return p, hist
    
    def __global_hiseq(self, img:np.ndarray, need_hist=False)->tuple:
        
        h0, histogram = self.__hiseq(img=img)
        m,n = img.shape
        img_hiseq = np.zeros((m*n), dtype=np.uint8)
        for i,pi in enumerate(img.flatten()):
            img_hiseq[i] = histogram[pi]
        
        img_hiseq = img_hiseq.reshape((m,n))
       
        if not need_hist:
            return img_hiseq
        
        return {
            'origin':h0.astype(np.uint16).tolist(), 
            'transform':self.__count(img_hiseq).astype(np.uint16).tolist()
        }, img_hiseq

    def __local_hiseq(self, img:np.ndarray,bsize:int, need_hist=False)->tuple:
    
        img_hiseq=np.copy(img)
        hist = []
        for i in range(0, img.shape[1], bsize):
            for j in range(0, img.shape[0], bsize):
                #print(f"{i}-{i+bsize}, {j}-{j+bsize}") 
                subimg = img[i:i+bsize, j:j+bsize]
                subimg_ = None
                if need_hist:
                    b_h0, subimg_=self.__global_hiseq(img=subimg, need_hist=need_hist)
                    hist.append(b_h0)
                else: 
                    subimg_=self.__global_hiseq(img=subimg, need_hist=need_hist)
                
                img_hiseq[i:i+bsize, j:j+bsize] = subimg_
                
        if not need_hist:
            return img_hiseq
        
        return {
            "blocks":hist, 
            "all":{
                'origin':self.__count(img).tolist(), 
                'transform':self.__count(img_hiseq).tolist()
            }
        }, img_hiseq

def makepath(p)->os.PathLike:
    if not os.path.exists(p):
        os.mkdir(p)
    return p

def histogramequalization(imgs_info:list, savedir=None, spatial="global", needresult=False, needhist=False, **kwarg):
    
    bar = tqdm(imgs_info)
    hist_eqer = GrayLevel_histogramEqualizer()
    hist = []
    result = []

    for img_info in bar:
        h = None
        bar.set_postfix_str(f"{spatial} hist eq for {img_info['name']}")
        if needhist:
            h ,img_ = hist_eqer.transform(img_info['img'], spatial, needhist, **kwarg)
            hist.append(h)
        else:
            img_ = hist_eqer.transform(img_info['img'], spatial,needhist, **kwarg)
        result.append(img_)
        bar.set_postfix_str(f"ok ..")
    
        if savedir is not None:
            
            thisimg_sp = makepath(os.path.join(savedir,f"{img_info['name']}"))
            
            imgsavepath = os.path.join(thisimg_sp,f"{img_info['name']}.{img_info['type']}")
            if not cv2.imwrite(imgsavepath, img_):
                print(f"problem for saving {img_info['name']}")
            
            if needhist:
                with open(os.path.join(thisimg_sp, "hist.json"), "w+") as jf:
                    json.dump(h,jf, indent=4, ensure_ascii=False)

    if needresult:
        if needhist:
            return result, hist
        return result
    else:
        if needhist:
            return hist

def hist_barchart(data, block:tuple, savepath, term)->None:

    fig = plt.figure(dpi=800)
    pid = 1
    gray = np.arange(256)
    for i in range(block[0]):
        for j in range(block[1]):
            ax0 = fig.add_subplot(block[0], block[1], pid)
            l1 =ax0.bar(gray, data[pid-1]['origin'], color='blue',alpha=0.5, label="ori")
            l2 =ax0.bar(gray, data[pid-1]['transform'], color='red',alpha=0.5, label="tra")
            pid+=1
        
    plt.legend([l1, l2],['origin','transform'], loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(savepath,f"histogram{term}.jpg"))
    plt.close()


if __name__ == "__main__":
    
    print("read images ..", end=" ", flush=True)
    p1 = cv2.imread(
        os.path.join("HW1_test_image","Lena.bmp"),
        cv2.IMREAD_GRAYSCALE
    )
    p2 = cv2.imread(
        os.path.join("HW1_test_image","Peppers.bmp"),
        cv2.IMREAD_GRAYSCALE
    )
    print("ok ..")

    imglist = [
        {'name':'Lena', 'img':p1,'type':"bmp"},
        {'name':'Peppers', 'img':p2,'type':"bmp"}
    ]

    resultpath = makepath(os.path.join(".", "result1"))

    global_savepath = makepath(os.path.join(resultpath, "globalhist")) 
    print("global histogram equlization")
    h_global = histogramequalization(imglist, global_savepath, "global", False, True)
    print("plotting histogram for global ..")
    for imgname, pi in zip(['Lena','Peppers'], h_global):
        print(imgname)
        hist_barchart(
            data=[pi],block=(1,1), 
            savepath=makepath(
                os.path.join(global_savepath,imgname)
            ), term=''
        )


    local_savepath = makepath(os.path.join(resultpath, "localhist"))
    bsize= p1.shape[0]//4 # equally 16 block 
    print("local histogram equlization")
    h_local = histogramequalization(imglist, local_savepath, "local", False,True, bsize=bsize)
    print("plotting histogram for local ..")
    for imgname, pi in zip(['Lena','Peppers'], h_local):
        print(imgname)
        hist_barchart(
            data=[pi['all']], block=(1,1),
            savepath=makepath(os.path.join(local_savepath,imgname)),
            term='all'
        )
        for i in tqdm(range(0, 16, 8)):
            hist_barchart(
                data=pi['blocks'][i:i+8],block=(2,4), 
                savepath=makepath(
                    os.path.join(local_savepath,imgname)
                ), term=i
            )