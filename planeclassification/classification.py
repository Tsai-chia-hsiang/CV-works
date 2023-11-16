import os
import argparse
import os.path as osp
import gc
from tqdm import tqdm
import pandas as pd
import cv2
from dataset import makedir, get_pure_name, Unlabel_Dataset, Labeled_Dataset, save_dict_as_csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--topk", help="k for knn", default=5, type=int)
parser.add_argument("-using_features",'--features', nargs='*', help="the folder for features of training and testing data")
parser.add_argument("--output",help="the directory for saving features", default=osp.join("features"))
args = parser.parse_args()


def euclidean_distance(x:np.ndarray, samples:np.ndarray)->np.ndarray: 
    return np.sum((samples-x)**2, tuple(range(samples.ndim-1, 0, -1)))


def KNN(x:np.ndarray, samples:np.ndarray, distance_finction=euclidean_distance, k:int=5 )->np.ndarray:
    ed = distance_finction(x=x, samples=samples)
    ordered = np.argsort(ed)
    return ordered[:k]

def imread(p:os.PathLike,size:tuple=(256,256), normalized=False)->np.ndarray:
    I =  cv2.resize(cv2.imread(p).astype(np.float32), size)
    if normalized:
        I = cv2.normalize(I, None)
    return I 

def vote(vote_box:np.ndarray)->int:
    u, c = np.unique(vote_box, return_counts = True) #count # of votes
    predict = u[c == c.max()][0] #max vote class
    return predict

def main():
    
    for f in args.features:
        result = {
            'file':[],
            'species':[]
        }
        
        fk = f if f != 'raw' else 'data'
        print(f"using : {fk} to calculate K-NN ")
        saveto = makedir(osp.join(makedir(args.output), f))
        read_method = np.load if f!= 'raw' else imread
        
        training_data = Labeled_Dataset(root=osp.join(fk, 'train'))
        testing_data = Unlabel_Dataset(root=osp.join(fk, 'test'))

        L, ALL_TRAIN = training_data.load_all(read_method=read_method,normalized=True)
        NAME, ALL_TEST = testing_data.load_all(read_method=read_method, normalized=True)

        print("calculating KNN for each testing sample : ")
        
        for i, fi in enumerate(tqdm(NAME)):

            y = vote(L[KNN(x=ALL_TEST[i], samples=ALL_TRAIN, k=args.topk)])
            result['file'].append(get_pure_name(fi, posfix_change='png'))
            result['species'].append(training_data._classes[y])
        
        save_dict_as_csv(d=result, path=osp.join(saveto, f"{f}_submission.csv"))
  
        del ALL_TRAIN, L
        gc.collect()

if __name__ == "__main__":
    main()