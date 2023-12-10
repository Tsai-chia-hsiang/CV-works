# GMM for AR face dataset

## Dataset:
10 people, each person has 13 facial images for training data and testing data.

## Training :
Appling GMM (E step and M step) for each person to get the parameters:

$(\alpha_b^{k},\mu_b^{k}, \Sigma_{b}^{k}), b\in\{1, 2, ... \text{number of people}\}; k\in\{1, 2, ..., K\}$ 

**K : number for gaussian component.

**number of people = 10

## Testing :
for one sample $x_j$

$$s_b^{x_j}=\displaystyle\sum_{i=1}^{K} P(b|x_j,\mu_b^i, \Sigma_b^i)$$

$P(b|x_j,\mu_b^i, \Sigma_b^i)=\alpha_b^i\times N(x_j|\mu_b^i, \Sigma_b^i)$

and the prediction class(person) of the sample $s$ is 

$$\hat{b}^{x_j} = \mathop{\arg\max}\limits_{b}(s^{x_j}_b), b\in \{1, 2, ... \text{number of people}\}$$

## To run the code:
using command : ```python gmm.py```
arguments:
- --max_train_iter : 
  - maximum of training iteration 
  - type : int
  - default=100
- --k : 
  - the component of GMM for each class
  - type : int
  - default = 2
- --repeate : 
  - the number for repeating the experiment
  - type : int
  - default = 1
- --dataroot:
  - the directory of data
  - default : ```./CSVdata/```
- --detail_log:
  - training log for each class in one experiment
  - default = False
- --saveto
  - the directory for saving parameters of the model
  - default : ```./model/```
  
e.g : ```python gmm.py --repeate 5 --k 2 --max_train_iter 100 --saveto ./model_v1/```

## Experiment result:
```python .\gmm.py --repeate 5 --k 2 --max_train_iter 800```

best accuracy : 0.85384613

- parameters : save at ```./bestmodel/```
  - Using ```np.load(osp.join("./bestmodel", "para.npy"), allow_pickle=True)``` to read.
  - Can use ```load_pretrained_parameters()``` to get the gmm model with the pretrained parameters.