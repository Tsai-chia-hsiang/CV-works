import os
import os.path as osp
import argparse
from tqdm import tqdm
import cv2
from dataset import Unlabel_Dataset, Labeled_Dataset, makedir, Dataset
from ImageFeatureExtraction.feature_extractor import *

parser = argparse.ArgumentParser()
parser.add_argument("--training_data",help=" the directory of training data", default="")
parser.add_argument("--testing_data",help=" the directory of testing data", default="")
parser.add_argument("-features",'--features', nargs='*', help="the method you want to use to extract the features")
parser.add_argument("--output",help="the directory for saving features", default=osp.join("features"))
args = parser.parse_args()

def extract(dataset:Dataset, saveroot:os.PathLike, extract_method:dict):
    
    for d in tqdm(dataset):
        
        path = d if not isinstance(d, tuple) else d[1]
        imgname = osp.split(path)[-1]
        I = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        for method in args.features:
    
            saveto = makedir(osp.join(saveroot,method))
            if isinstance(d, tuple):
                saveto = makedir(osp.join(saveto, dataset._classes[d[0]]))
            
            extract_method[method](
                I, need_return=False, saveto=osp.join(saveto, imgname[:-4])
            )
    

def main(method:dict):
    
    if len(args.training_data):

        print(args.training_data)
        training_data = Labeled_Dataset(root=args.training_data)
        training_feature_dir = makedir( osp.join(makedir(args.output), "train") )
        print(training_feature_dir)

        extract(
            dataset=training_data, 
            saveroot=training_feature_dir, 
            extract_method=method
        )

    if len(args.testing_data):

        print(args.testing_data)
        testing_data = Unlabel_Dataset(root=args.testing_data)
        testing_feature_dir = makedir( osp.join(makedir(args.output), "test") )
        print(testing_feature_dir)
        extract(
            dataset=testing_data, 
            saveroot=testing_feature_dir,
            extract_method=method
        )

if __name__ == "__main__":

    feature_extraction_map = {}
    
    if 'gabor' in args.features:
        feature_extraction_map['gabor'] = Gabor_Features_Extractor(sigma=1.4,Nangle=12)
    
    if 'colorhistogram' in args.features:
        feature_extraction_map['colorhistogram'] = Color_Histogram()
        
    if 'hog' in args.features:
        feature_extraction_map['hog'] = HoG()
    
    
    main(method=feature_extraction_map)