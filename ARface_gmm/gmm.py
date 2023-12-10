import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--max_train_iter",help="maximum of training iteration", type=int, default=100)
parser.add_argument("--k",help="the component of GMM for each class", type=int, default=2)
parser.add_argument("--detail_log",help="training log for each class in one experiment", type=bool, default=False)
parser.add_argument("--repeate",help="the number for repeating the experiment", type=int, default=1)
parser.add_argument("--dataroot",help="the directory of data", default=osp.join("CSVdata"))
args = parser.parse_args()

def getdata(root:os.PathLike)->dict:
    
    data = {
        'train':{
            "data":np.loadtxt(osp.join(root, "TrainData.csv"),delimiter=","),
            "label":np.loadtxt(osp.join(root, "TrainLabel.csv"),delimiter=",",dtype=np.int32)
        },
        'test':{
            'data':np.loadtxt(osp.join(root, "TestData.csv"),delimiter=","),
            'label':np.loadtxt(osp.join(root, "TestLabel.csv"),delimiter=",", dtype=np.int32)
        }
        
    }
    classes = np.unique(data['train']['label'])
    for k in list(data.keys()):
        data[k]['data'] = data[k]['data'][:, :classes.shape[0]]

    return data, classes

def covariance_matrix(x:np.ndarray)->np.ndarray:
    shift = x - np.repeat(np.mean(x, axis=0, keepdims=True), x.shape[0], axis=0)
    return (shift.T@shift)/(x.shape[0]-1)

def gaussian_pdf(x:np.ndarray, mean:np.ndarray, cov:np.ndarray, perturbation:bool=False)->np.ndarray:
        
    def d_mahalanobis(x:np.ndarray, mean:np.ndarray, invcov:np.ndarray):
        shift = x - np.repeat(mean.reshape(1, -1), x.shape[0], axis=0)
        return np.sum((shift@invcov)*shift, axis=1)
        
    pcov = cov if not perturbation else cov+np.eye(cov.shape[0])*3000000
    score = np.exp(-0.5*d_mahalanobis(x, mean=mean, invcov=np.linalg.inv(pcov)))
    normalze = (((2*np.pi)**x.shape[1]) * (np.linalg.det(pcov)))**0.5

    return score/normalze


class Labeled_GMM():

    def __init__(self, num_classes:int, k:int) -> None:

        self.num_classes = num_classes
        self.k = k
        self.mean:list[list[np.ndarray]]=[[None for _ in range(k)] for i in range(num_classes)]
        self.cov:list[list[np.ndarray]]=[[None for _ in range(k)] for i in range(num_classes)]
        self.alpha:list[np.ndarray]=[None for _ in range(num_classes)]
        
    def train(self, data:np.ndarray, label:np.ndarray, max_iter:int=200, train_log:bool=True)->None:
        
        class_num_bar = range(self.num_classes)
        if not train_log:
            class_num_bar = tqdm(class_num_bar)
        
        for li in class_num_bar:
            if train_log:
                print(li)
            self.__trian_one_class_gmm_model(
                x=data[np.where(label == li)[0]],
                ci=li, max_iter=max_iter, 
                train_log=train_log
            )

    def __inital_parameters(self, x:np.ndarray, ci:int)->None:

        random_sample_idx = np.random.choice(np.arange(x.shape[0]), self.k, replace=False)
        covx = covariance_matrix(x)

        for idx, ki in zip(random_sample_idx, range(self.k)):
            self.mean[ci][ki] = x[idx:idx+1, :]
            self.cov[ci][ki] = covx
        
        self.alpha[ci] = np.ones((1,self.k))/self.k
    
    def __weighted_pdf_k_compoment(self, x:np.ndarray, ci:int)->np.ndarray:
    
        pdf = np.hstack(
                [
                    gaussian_pdf(
                        x=x, mean=self.mean[ci][ki], cov=self.cov[ci][ki],
                        perturbation=True
                    ).reshape(-1, 1) for ki in range(self.k)
                ]
            )

        pdf = self.alpha[ci]*pdf
        return pdf

    def __trian_one_class_gmm_model(self, x:np.ndarray, ci:int, max_iter:int=200, train_log:bool=True)->None:
        
        self.__inital_parameters(x, ci)
        pbar = range(max_iter)
        if train_log:
            pbar = tqdm(pbar)

        for _ in pbar:

            pdf = self.__weighted_pdf_k_compoment(x=x, ci=ci)
            pdf = pdf/(np.sum(pdf, axis=1).reshape(-1,1))
      
            for ki in range(self.k):

                cluster_pdf = pdf[:, ki].reshape(-1,1)
                pdf_summa = np.sum(cluster_pdf)
                
                self.mean[ci][ki] = np.sum(cluster_pdf*x,axis=0, keepdims=True)/pdf_summa
   
                shift = x - np.repeat(self.mean[ci][ki], x.shape[0] ,axis=0)
                self.cov[ci][ki] = shift.T@(cluster_pdf*shift)/pdf_summa
            
            self.alpha[ci] = np.mean(pdf, axis=0, keepdims=True)

    def predict(self, x:np.ndarray)->np.ndarray:
        score = np.hstack(
            [np.sum(self.__weighted_pdf_k_compoment(x=x, ci=ci), axis=1).reshape(-1, 1) 
            for ci in range(self.num_classes)]
        )
        l = np.argmax(score, axis=1)
        return l

    def save(self, saveroot:os.PathLike)->None:
        
        if not osp.exists(saveroot):
            os.mkdir(saveroot)
        
        with open(osp.join(saveroot, "properties.json"),"w+", encoding="utf-8") as jf:
            json.dump(
                {'k':self.k, 'num_class':self.num_classes}, 
                jf, ensure_ascii=False, indent=4
            )
            np.save(osp.join(saveroot, "mean"),np.array(self.mean, dtype=np.object_),allow_pickle=True)
            np.save(osp.join(saveroot, "cov"),np.array(self.cov, dtype=np.object_),allow_pickle=True)
            np.save(osp.join(saveroot, "alpha"),np.array(self.alpha, dtype=np.object_),allow_pickle=True)

    def load(self, saveroot:os.PathLike)->None:
        m = np.load(osp.join(saveroot, "mean.npy"), allow_pickle=True)
        cov = np.load(osp.join(saveroot, "cov.npy"), allow_pickle=True)
        a = np.load(osp.join(saveroot, "alpha.npy"), allow_pickle=True)

        for ci in range(self.num_classes):
            for ki in range(self.k):
                self.mean[ci][ki] = m[ci][ki].astype(np.float64)
                self.cov[ci][ki] = cov[ci][ki].astype(np.float64)
            self.alpha[ci] = a[ci].astype(np.float64)


def load_pretrained_parameters(saveroot:os.PathLike)->Labeled_GMM:
    modelprperty = {}
    with open(osp.join(saveroot, "properties.json")) as p:
        modelprperty = json.load(p)
    gmm = Labeled_GMM(num_classes=modelprperty['num_class'], k=modelprperty['k'])
    gmm.load(saveroot=osp.join(saveroot))
    return gmm


def main():

    data,classes = getdata(root=args.dataroot)
    acc_list = []
    acc_best = 0.0
    for _ in range(args.repeate):
        
        gmm = Labeled_GMM(num_classes=len(classes), k=args.k)
        
        gmm.train(
            data=data['train']['data'], label=data['train']['label'], 
            max_iter=args.max_train_iter,train_log=args.detail_log
        )

        predict = gmm.predict(x=data['test']['data'])
        acc = np.mean((predict == data['test']['label']).astype(np.float32))
        if acc > acc_best:
            acc_best = acc
            gmm.save(saveroot=osp.join("model"))
        print(f"accuracy : {acc:.3f}")
        acc_list.append(acc)
    print(f"mean accuracy over {args.repeate} times experiments : {np.mean(acc_list):.3f}")
   

if __name__ == "__main__":

    main()
    #data,classes = getdata(root=args.dataroot)
    #gmm = load_pretrained_parameters(osp.join("bestmodel"))
    #predict = gmm.predict(x=data['test']['data'])
    #acc = np.mean((predict == data['test']['label']).astype(np.float32))
    #print(acc)