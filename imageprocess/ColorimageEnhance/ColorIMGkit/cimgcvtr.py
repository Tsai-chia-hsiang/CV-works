import numpy as np
from tqdm import tqdm

class SupportDomain:
    rgb = 0
    hsi = 1
    lab = 2

supportdomain = SupportDomain()

class DomainConvertor():
    
    def __init__(self) -> None:

        self.__eps = np.float64(1e-17)

        self.__Mrgb2xyz = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ], dtype=np.float64
        )

        self.__Xw = 0.950467
        self.__Yw = 1.0
        self.__Zw = 1.088969
        
        self.__Mxyz2rgb = np.array(
            [
                [ 3.24045484, -1.53713885, -0.49853155],
                [-0.96926639,  1.87601093,  0.04155608],
                [ 0.05564342, -0.20402585,  1.05722516]
            ], dtype=np.float64)

        self.__convertor = [
            [0, self.__rgb2hsi, self.__rgb2lab],
            [self.__hsi2rgb, 0, -1],
            [self.__lab2rgb, -1, 0]
        ]
 
    def convert(self, img, fromD, toD)->np.ndarray:
        cvtimg = None

        if self.__convertor[fromD][toD] == 0:
            cvtimg = img.copy()
        
        elif self.__convertor[fromD][toD] == -1:
            cvtimg = self.__convertor[0][toD](
                self.__convertor[fromD][0](img)
            )
        
        else:
            cvtimg = self.__convertor[fromD][toD](img)
        
        return cvtimg
    
    def __rgb2hsi(self, img:np.ndarray)->np.ndarray:

        rgb = img.astype(np.float64) / 255.0
        # 0: R, 1:G, 2:B
        R, G, B = 0, 1, 2
        DRG = rgb[..., R] - rgb[..., G]
        DRB = rgb[..., R] - rgb[..., B]
        DGB = rgb[..., G] - rgb[..., B]
        RGBSUM = np.sum(rgb, axis=2)
        shiftphasoridx = np.where(DGB < 0)
        as_zero_theta = np.where((DRG**2 + (DRB) * (DGB)) <= self.__eps)
        H = np.arccos(
            0.5 * (DRG + DRB) / (
                (DRG**2 + (DRB) * (DGB)) ** 0.5 + self.__eps
            )
        )
        H[shiftphasoridx] = 2 * np.pi - H[shiftphasoridx]
        H[as_zero_theta] = 0
        I = (RGBSUM) / 3
        S = 1 - (3*np.min(rgb, axis=2) / (RGBSUM + self.__eps))
        
        return np.dstack((H, S, I))

    def __hsi2rgb(self, img:np.ndarray)->np.ndarray:
            
        def cvt(h0,s,i, theta):
            h = h0 - theta
            C0 = i*(1-s)
            C1 = i*(1+s*np.cos(h)/np.cos(np.pi/3 - h))
            C2 = 3*i - (C0 + C1)
            return (C0, C1, C2)
            
        H = img[..., 0]
        S = img[..., 1]
        I = img[..., 2]
        R = np.zeros(H.shape)
        B = R.copy()
        G = R.copy()
        theta_upper = np.arange(1, 4)
        theta_lower = np.arange(3)
        for l, u in zip(theta_lower, theta_upper):
            mask = None
            ltheta =l*2*np.pi/3
            utheta =u*2*np.pi/3
            if u == 3:
                mask = np.where((H>=ltheta) & (H<=utheta))
            else:
                mask = np.where((H>=ltheta) & (H<utheta)) 
            C0, C1, C2 = cvt(H[mask], S[mask], I[mask], theta=ltheta)
            
            if l == 0:
                R[mask],G[mask],B[mask] = C1, C2, C0
            elif l == 1:
                R[mask],G[mask],B[mask] = C0, C1, C2
            elif l == 2:
                R[mask],G[mask],B[mask] = C2, C0, C1
        
        return np.dstack(
             [np.clip(R*255, 0 ,255).astype(np.uint8),
              np.clip(G*255, 0 ,255).astype(np.uint8),
              np.clip(B*255, 0 ,255).astype(np.uint8)]
            )
    
    def __rgb2xyz(self, img:np.ndarray)->np.ndarray:
        return (img.astype(np.float64)/255.0)@self.__Mrgb2xyz.T
    
    def __xyz2rgb(self, img:np.ndarray)->np.ndarray:
        return np.clip((img@self.__Mxyz2rgb.T)*255, 0, 255).astype(np.uint8)
    
    def __xyz2lab(self, img:np.ndarray)->np.ndarray:
        def h(q:np.ndarray)->np.ndarray:
            cub_sqrt_idx = np.where(q > 0.008856)
            other = np.where(q <= 0.008856)
            hq = np.zeros(q.shape, dtype=np.float64)
            hq[cub_sqrt_idx] = q[cub_sqrt_idx]**(1/3)
            hq[other] = 7.787*q[other] + 16/116
            return hq
        hx = h(img[..., 0]/self.__Xw)
        hy = h(img[..., 1]/self.__Yw)
        hz = h(img[..., 2]/self.__Zw)
        L = 116.0*hy-16.0
        a = 500.0*(hx-hy)
        b = 200.0*(hy-hz)
        return np.dstack([L, a, b])

    def __lab2xyz(self, img:np.ndarray)->np.ndarray:
        
        def h(q:np.ndarray, w)->np.ndarray:
            cub_idx = np.where(q >= 0.008856)
            other = np.where(q < 0.008856)
            hq = np.zeros(q.shape, dtype=np.float64)
            hq[cub_idx] = (q[cub_idx]**3)*w
            hq[other] = ((q[other]-16)/116)*3*(0.008856**2)*w
            return hq
        
        fy = (img[..., 0]+16)/116
        fx = (img[..., 1]/500)+fy
        fz = fy - (img[..., 2]/200)
    
        return np.dstack([h(fx, self.__Xw), h(fy, self.__Yw), h(fz,self.__Zw)])
    
    def __rgb2lab(self, img:np.ndarray)->np.ndarray:
        
        xyz = self.__rgb2xyz(img)
        return self.__xyz2lab(xyz)

    def __lab2rgb(self, img:np.ndarray)->np.ndarray:
        
        xyz = self.__lab2xyz(img)
        return self.__xyz2rgb(xyz)
    

