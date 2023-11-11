import os
import os.path as osp
def walkdir(root:os.PathLike)->list:
    ret = []
    for r,_,fs in os.walk(root):
        for f in fs:
            ret.append(osp.join(r, f))
        return ret


class Dataset():
    def __init__(self, root :os.PathLike, shuffle:bool=False) -> None:
        
        self._classes, self._datapathes = Dataset.walk_data_root(root=root)

    @staticmethod
    def walk_data_root(root:os.PathLike)->tuple:
    
        c = [c.name for c in os.scandir(root)]
        ret = []
        for ciid, ci in enumerate(c):
            class_dir = osp.join(root, ci)
            x = walkdir(root= class_dir)
            ret += [(ciid, xi) for xi in x]
        return c, ret

    def __getitem__(self, i):
        return self._datapathes[i]
    def __len__(self):
        return len(self._datapathes)