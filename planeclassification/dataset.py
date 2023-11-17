import os
import os.path as osp
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def walkdir(root:os.PathLike)->list:
    ret = []
    for r,_,fs in os.walk(root):
        for f in fs:
            ret.append(osp.join(r, f))
    return ret

def makedir(p:os.PathLike)->os.PathLike:
    if not osp.exists(p):
        os.mkdir(p)
    return p

def get_pure_name(f, posfix_change:str=None)->str:
    n = osp.split(f)[-1]
    if posfix_change is not None:
        n = n[:n.rfind('.')+1]+posfix_change
    return n

def save_dict_as_csv(d:dict, path:os.PathLike):
    df = pd.DataFrame(d)
    df.to_csv(path, index=False)



class Dataset():

    def __init__(self, root:os.PathLike) -> None:
        self._datapathes = self.walk_data_root(root=root)

    def walk_data_root(self, root:os.PathLike) -> list:
        raise NotImplementedError
    
    def __getitem__(self, i):
        return self._datapathes[i]

    def __len__(self):
        return len(self._datapathes)
    
    def get_all(self)->list:
        return self._datapathes.copy()

class Unlabel_Dataset(Dataset):

    def __init__(self,root :os.PathLike) -> None:
        super().__init__(root=root)    
    
    def walk_data_root(self, root:os.PathLike) -> list:
        return walkdir(root)
    
class Labeled_Dataset(Dataset):
    
    def __init__(self, root: os.PathLike) -> None:
        super().__init__(root)
        self._classes , self._datapathes = self._datapathes[0], self._datapathes[1]
    
    def walk_data_root(self, root:os.PathLike)->tuple:
    
        c = [c.name for c in os.scandir(root)]
        ret = []
        for ciid, ci in enumerate(c):
            class_dir = osp.join(root, ci)
            x = walkdir(root= class_dir)
            ret += [(ciid, xi) for xi in x]
        return c, ret

    def get_all(self)->tuple:
        labels, p = zip(*self._datapathes)
        labels = np.array(labels)
        return labels, list(p)

class Data_Loader():

    def __init__(self) -> None:
        self.ftype = [['.npy'], ['.png', '.jpg']]
        self.input_method = [np.load, cv2.imread]

    def load_single(self, p:os.PathLike, preprocess=lambda x:x, **kwargs)->np.ndarray:
        ftype = p[p.rfind('.'):]
        i = 0
        for i, fi in enumerate(self.ftype):
            if ftype in fi:
                break
        I = self.input_method[i](p)
        return preprocess(I, **kwargs)   

    def __call__(self, p:os.PathLike|list, preprocess=lambda x:x, **kwargs)->np.ndarray:

        if isinstance(p, str):
            return self.load_single(p, preprocess=preprocess, **kwargs)
        
        elif isinstance(p, list):
            return np.stack(
                [self.load_single(pi, preprocess=preprocess, **kwargs) for pi in tqdm(p)]
            )
        else:
            raise ValueError
