## Assigment 2 Report 



## Extraction :
command : ```python extract.py --training_data /path/to/trainingdatafolder/ --testing_data /path/to/testingdatafolder/ -features extraction method you want to apply --output /path/to/output```

- --training_data/--testing_data : folder where training/testing data are at.
  - if you don't want to extract training data, it is optional argument and just don't pass it. 
- -features
  - The extraction methods you want to apply.
  - a list, can pass multi method in a row.
  - currently support:
    - HoG (hog)
    - Color Histogram (colorhistogram)
    - Gabor Filter (gabor)
- --output:
  - the root for saving the features that be extracted.

e.g.
```python extract.py --training_data ./data/train/ -features gabor colorhistogram hog  --output ./features/```


## classification :

command : ```python classification.py --feature_root folder where features be saved at --rawdata the folder where raw data is at -using_features feature you want to use  --output /path/to/output```

e.g. 
```python classification.py --feature_root ./feature --rawdata ./data -using_features raw colorhistogram gabor hog  --output ./result```


Kaggle submission :
<img src="./result/score.png">


### Reference :
- HoG : 
  - https://www.youtube.com/watch?v=5nZGnYPyKLU
  - https://hannibunny.github.io/orbook/features/HoGfeatures.html#normalization