import argparse
import os.path as osp
import gc
from tqdm import tqdm
import cv2
from dataset import makedir, get_pure_name, Unlabel_Dataset, Labeled_Dataset, Data_Loader, save_dict_as_csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--topk", help="k for knn", default=5, type=int)
parser.add_argument("-using_features",'--features', nargs='*', help="the folder name of the features of training and testing data")
parser.add_argument("--feature_root",help="the directory for saving features", default=osp.join("features"))
parser.add_argument("--rawdata",help="the directory for raw data", default=osp.join("data"))
parser.add_argument("--output",help="the directory for saving features", default=osp.join("result"))
args = parser.parse_args()

def euclidean_distance(x:np.ndarray, samples:np.ndarray)->np.ndarray: 
    return np.sum((samples-x)**2, tuple(range(samples.ndim-1, 0, -1)))

class KNN():
    
    def __init__(self) -> None:
        pass
    
    def __get_simi(self, x:np.ndarray, samples:np.ndarray, distance_function=euclidean_distance)->np.ndarray:
       
        d = distance_function(x=x, samples=samples)
        return np.argsort(d)

    def __call__(
        self, x:np.ndarray, samples:np.ndarray, 
        samples_label:np.ndarray, classes_index:list,
        distance_function=euclidean_distance, topk=5
    ):
        
        kth_nn = self.__get_simi(
            x=x, samples=samples, 
            distance_function=distance_function
        )[:topk]
        kth_nn_id = samples_label[kth_nn]
        popularity = self.__vote(kth_nn_id)
        return classes_index[popularity]
    
    def __vote(self, vote_box:np.ndarray)->int:

        u, c = np.unique(vote_box, return_counts = True) #count # of votes
        predict = u[c == c.max()][0] #max vote class
        return predict


def impreprocess(img:np.ndarray,s:tuple, dtype:np.dtype=np.float32, scale=False)->np.ndarray:
    I = cv2.resize(img, s)
    I = I.astype(dtype=dtype)
    if scale:
        I /= 255.0
    return I



preprocess_map = {
    'raw':{
        'method':impreprocess,
        'kwargs':{'s':(256,256), 'dtype':np.float32, 'scale':True}
    },
    'gabor':{
        'method':impreprocess,
        'kwargs':{'s':(256,256), 'dtype':np.float32, 'scale':True}
    },
    'hog':{
        'method':lambda x:x,
        'kwargs':{}
    },
    'colorhistogram':{
        'method': lambda x:x.astype(np.float32),
        'kwargs':{}
    }
}

def main():

    knn_predictor = KNN()
    data_loader = Data_Loader()

    for f in args.features:
        result = {'file':[],'species':[]}
        train_path = osp.join(args.feature_root, 'train',f) if f != 'raw' else osp.join(args.rawdata, 'train')
        test_path = osp.join(args.feature_root, 'test', f) if f != 'raw' else osp.join(args.rawdata, 'test')

        print(f"using : {f} to calculate K-NN : {train_path} , {test_path} ")
        
        training_data = Labeled_Dataset(root=train_path)
        testing_data = Unlabel_Dataset(root=test_path)

        L, all_train_files = training_data.get_all()
        all_test_files = testing_data.get_all()

        ALL_TRAIN = data_loader(
            all_train_files, 
            preprocess=preprocess_map[f]['method'], 
            **preprocess_map[f]['kwargs']
        )
        ALL_TEST = data_loader(
            all_test_files,
            preprocess=preprocess_map[f]['method'], 
            **preprocess_map[f]['kwargs']
        )
        
        print("calculating KNN for each testing sample : ")
    
        for i, fi in enumerate(tqdm(all_test_files)):

            p = knn_predictor(
                x=ALL_TEST[i], samples=ALL_TRAIN, 
                samples_label=L, classes_index=training_data._classes,
                topk=args.topk
            )
            result['file'].append(get_pure_name(fi, posfix_change='png'))
            result['species'].append(p)
        
        save_dict_as_csv(
            d=result, 
            path=osp.join(makedir(args.output), f"{f}_submission.csv")
        )
  
        del ALL_TRAIN, L
        gc.collect()

if __name__ == "__main__":
    main()