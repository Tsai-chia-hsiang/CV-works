import os
import os.path as osp
import pandas as pd
import numpy as np
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
    
    def load_all(self, read_method, **kwargs)->tuple:
        raise NotImplementedError


class Unlabel_Dataset(Dataset):

    def __init__(self,root :os.PathLike) -> None:
        super().__init__(root=root)    
    
    def walk_data_root(self, root:os.PathLike) -> list:
        return walkdir(root)
    
    
    def load_all(self, read_method, **kwargs)->tuple:
        """
        Warning ! will load all data into memory, please note 
        the size of the dataset
        """
        print('loading data: ')
        return self._datapathes.copy(), np.stack(
            [read_method(p, **kwargs) for p in tqdm(self._datapathes)]
        )

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

    def load_all(self, read_method, **kwargs)->tuple:
        """
        Warning ! will load all data into memory, please note 
        the size of the dataset
        """
        print("loading data : ")
        labels, p = zip(*self._datapathes)
        labels = np.array(labels)
        x = np.stack([read_method(pi, **kwargs) for pi in tqdm(p)])
        return labels, x

    
