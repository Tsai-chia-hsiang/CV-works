import os
import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from feature_extraction import Gabor_Filter,  HoG, color_histogram
from dataset import walkdir, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--training_data",help=" the directory of training data", default="")
parser.add_argument("--testing_data",help=" the directory of testing data", default="")
parser.add_argument("-features",'--features', nargs='*', help="the method you want to use to extract the features")
parser.add_argument("--output",help="the directory for saving features", default=osp.join("features"))
args = parser.parse_args()

def makedir(p:os.PathLike)->os.PathLike:
    if not osp.exists(p):
        os.mkdir(p)
    return p

gaborfilter = Gabor_Filter(sigma=1.4,Nangle=12)


feature_extraction_map = {
    'gabor':{
        'preprocess':lambda x:x,
        'function':gaborfilter.filt,
        'save':cv2.imwrite,
        'postprocess':lambda x:np.clip(x, 0, 255).astype(np.uint8)
    },
    'colorhistogram':{
        'preprocess':lambda x:x,
        'function':color_histogram,
        'save':np.save,
        'postprocess':lambda x:x
    },
    'hog':{
        'preprocess':lambda x:x,
        'function':HoG,
        'save':np.save,
        'postprocess':lambda x:x
    }
}

def extract(dataset:Dataset|list, saveroot:os.PathLike):
    
    for d in tqdm(dataset):
        
        path = d if not isinstance(d, tuple) else d[1]

        color_img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        imgname = osp.split(path)[-1]

        for method in args.features:

            I = color_img if method == 'colorhistogram' else gray_img
            
            saveto = makedir(osp.join(saveroot,method))
            if isinstance(d, tuple):
                saveto = makedir(osp.join(saveto, dataset._classes[d[0]]))

            savename = osp.join(saveto, imgname) if method == "garbor" else osp.join(saveto, imgname[:-4])
            
            out = feature_extraction_map[method]['function'](
                feature_extraction_map[method]['preprocess'](I), 
            )

            feature_extraction_map[method]['save'](
                savename, 
                feature_extraction_map[method]['postprocess'](out)
            )
    

def main():
    
    if len(args.training_data):

        print(args.training_data)
        training_data = Dataset(root=args.training_data)
        training_feature_dir = makedir( 
            osp.join(makedir(args.output), "train")
        )
        print(training_feature_dir)

        extract(dataset=training_data, saveroot=training_feature_dir)

    if len(args.testing_data):

        print(args.testing_data)
        testing_data = walkdir(root=args.testing_data)
        testing_feature_dir = makedir( 
            osp.join(makedir(args.output), "test")
        )
        print(testing_feature_dir)
        extract(dataset=testing_data, saveroot=testing_feature_dir)

if __name__ == "__main__":
    main()