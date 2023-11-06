import os
import cv2
from tqdm import tqdm

class rgbimg:
    def __init__(self, img, name) -> None:
        self.name = name
        self.img = img
    

def readimg(imgpath, name=None)->rgbimg:
    imgname=name
    if imgname is None:
        imgname = os.path.split(imgpath)[1]
    
    return rgbimg(
        name=imgname, 
        img=cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    )

def readimgs(imgdir)->list:
    imgs = []
    for r, _, fs in os.walk(imgdir):
        for f in tqdm(fs):
            p = os.path.join(r, f)
            imgs.append(readimg(imgpath=p, name=f))
            
    return imgs

def saveimg(img:rgbimg, savepath):
    cv2.imwrite(
        os.path.join(savepath, img.name), 
        cv2.cvtColor(img.img, cv2.COLOR_RGB2BGR)
    ) 

def saveimgs(imgs:list, savepath):
    print(f"save at {savepath}")
    for img in tqdm(imgs):
        saveimg(img, savepath = savepath)
