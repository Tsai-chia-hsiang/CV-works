import os
import numpy as np
from tqdm import tqdm
from ColorIMGkit.cimgcvtr import DomainConvertor, supportdomain
from ColorIMGkit.histEQ import ColorHistEQ
from ColorIMGkit.sharpening import RGBSharpener
from ColorIMGkit.RGBimgIO import readimgs, saveimgs, rgbimg
from ColorIMGkit.SaturationADJ import SaturationADJ, gamma_correction

rgb = supportdomain.rgb
hsi = supportdomain.hsi
lab = supportdomain.lab
domain_code= {0:"rgb", 1:"hsi", 2:"lab"}

convertor = DomainConvertor()
histeq = ColorHistEQ()
rgbsharpener = RGBSharpener()


def makepath(p):
    if not os.path.exists(p):
        os.mkdir(p)
    return p

def Batch_imgs_processing(imgs, apply_method, apply_domain, given_domain=rgb, autosave=None, *arg, **kwarg)->list:
    
    ret = []
    pbar = tqdm(imgs)
    for img in pbar:
        pbar.set_postfix_str(f"{img.name}")
        
        applyon = convertor.convert(
            img=img.img, fromD=given_domain, 
            toD=apply_domain
        )
        
        processed_img =  apply_method(applyon, *arg, **kwarg)
        
        processed_rgbimg = convertor.convert(
            img=processed_img, fromD=apply_domain, 
            toD=rgb
        )
        ret.append(rgbimg(img = processed_rgbimg, name=img.name))
    
    if autosave is not None:
        saveimgs(imgs=ret, savepath=autosave)
    
    return ret


def test_domain_converting(imgs, wanted_d=[hsi, lab]):
      
    testdir= makepath(os.path.join("testing1"))
    
    for d in wanted_d:
        Batch_imgs_processing(
            imgs,lambda x:x, d, rgb,
            makepath(os.path.join(testdir,domain_code[d]))
        )


def main(imgs):

    resultdir = makepath(os.path.join("result"))
    
  
    print("HistEQ on RGB")
    _ = Batch_imgs_processing(
        imgs, histeq.transform, 
        rgb, rgb, makepath(os.path.join(resultdir,"HistEQ_RGB")),
        "global", rgb, 
    )
    print("="*50)

    print("HistEQ on I of HSI")
    _ = Batch_imgs_processing(
        imgs, histeq.transform, 
        hsi, rgb, makepath(os.path.join(resultdir,"HistEQ_I")),
        "global", hsi 
    )
    print("="*50)


    print("Saturation gamma correction")
    sadj = Batch_imgs_processing(
        imgs,SaturationADJ,hsi,rgb, 
        makepath(os.path.join(resultdir,"Gamma")),
        gamma_correction,1,0.5
    )
    print("="*50)


    print("Hist on HSI with Saturation gamma correction")
    _ = Batch_imgs_processing(
        sadj, histeq.transform, 
        hsi, rgb, makepath(os.path.join(resultdir,"HistEQ_I_Ga")),
        "global", hsi
    )
    print("="*50)

    print("Sharpening on RGB")
    sharpening = Batch_imgs_processing(
        imgs, rgbsharpener.Laplacian_Sharpening, 
        rgb, rgb, makepath(os.path.join(resultdir,"RGBsharpening")),
        False
    )
    print("="*50)
    
    print("Sharpening on RGB & Hist on I of HSI")
    _ = Batch_imgs_processing(
        sharpening, histeq.transform, 
        hsi, rgb, makepath(os.path.join(resultdir,"HistEQ_N_sharp")),
        "global", hsi
    )
    print("="*50)
    

    print("HistEQ on L of L*a*b*")
    _ = Batch_imgs_processing(
        imgs, histeq.transform, 
        lab, rgb, makepath(os.path.join(resultdir,"HistEQ_Lab")),
        "global", lab
    )
    print("="*50)


if __name__ == "__main__":

    imgs = readimgs(imgdir=os.path.join("HW3_test_image"))
    
    #test_domain_converting(imgs=imgs)
    
    main(imgs=imgs)   
     
    
