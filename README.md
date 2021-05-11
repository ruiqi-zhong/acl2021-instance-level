## Are Larger Pretrained Language Models Uniformly Better? Comparing Performance at the Instance Level, ACL 2021 Findings

Authors: Ruiqi Zhong, Dhruba Ghosh, Dan Klein, Jacob Steinhardt

This GitHub Repo contains the pretrained models, model predictions (data) and code for the experiments.
If you have any questions, need additional information about the models/data, or want to request a new feature, feel free to send an email to ruiqi-zhong@berkeley.edu .

The paper will be public soon. 

### 1. Pre-trained Models

You can download all our pre-trained BERT models from [here](https://drive.google.com/drive/folders/1--niMIJNd3iMzc4UENZUc_MFXGdzLPDz?usp=sharing). 
We experimented with 5 different sizes (mini, small, medium, base, large) as described in the [official BERT github Repo](https://github.com/google-research/bert). 
Our models are also stored in the same format.

The directory for the medium model with pre-training seed 3 after being pre-trained for 2000000 steps is ```pretrained/medium/pretrain_seed3step2000000```.
We used the exact same pre-training code as BERT and similar training settings, except that we used the training corpus from , reduced the context size from 512 to 128, and increase the number of training steps from 1M to 2M. 
 
 ### 2. Model Predictions
 
We release the model predictions on 3 different tasks (sst-2, MNLI, and QQP) for 5 different model sizes, each with 10 pretraining seeds and 5 finetuning seeds, as described in the paper. 
Additionally, we provide the model predictions for several out-of-domain distributions (e.g. SNLI/HANS for MNLI, TwitterPPDB for QQP), and predictions at the 3.0, 3.33, 3.67 training epoch (in the paper we always finetune for 4 epochs).
You can download them from [here](https://drive.google.com/drive/folders/1jMqFE8SekJIjVIYGgoP5dIarz4JcWMmC?usp=sharing).

#### 2.1 model_data_for_release/
The datapoints and the raw probability predictions can be seen in the ```model_data_for_release/``` folder. 
For each task of MNLI, QQP and SST-2, it has the following files/folders (we performed 5 fold cross-validation on SST-2):

```data.json``` contains the data used for fine-tuning (training) and prediction, where each datapoint is represented as a dictionary. 
For example,

```buildoutcfg
import json
data = json.load(open('qqp/data.json'))
print('Number of datapoints for model prediction', len(data['predict'])) # prints the number of datapoints that are used for evaluation
print(data['predict'][0]) # prints the first datapoint for evaluation
```
, and we have

```buildoutcfg
Number of datapoints for model prediction, 79497
{'guid': 'dev-0', 'label': '1', 
'text_b': 'Where are the best places to eat in New York City?', 
'text_a': 'Where are the best places to eat in New York City that have a great vibe?'}
```

Notice that string before "-" for "guid" denotes the original data split, not our train/test split.
For example, "train" might occur in the evaluation set, since we used part of the original training split for testing.
In our paper, we used the "dev_matched" split for MNLI, "dev" for QQP, and "train" for SST-2. 
We downloaded our QQP data from [here](https://github.com/shreydesai/calibration), and the NLI challenge set from [here](https://github.com/owenzx/InstabilityAnalysis).


```[predict/train].tf_record``` contains the tokenized data in the tensorflow format that are used for fine-tuning. 

```size2hyperparam.json``` contains the hyper-parameter used for different model sizes. 

```results``` is the folder that contains the models' prediction, each in a .tsv format, representing a matrix of dimension (number of datapoints, number of classes).
Each row is the models' predicted probability for each class, and correspond to one datapoint in ```data['predict']```. 
```slargep9f5epoch9over3.tsv``` means the predictions of the large size model with pretraining seed 9 finetuning seed 5 evaluated at (9/3)=3 epoch.

#### 2.2 Processed Predictions: Correctness tensors

We extract 3 types of "correctness tensors" from the .tsv and data.json files, with the command

```python3 dump_correctness.py```

The results are dumped into ```correctness/```, ```correctness_p/```, ```ensemble_c/```


```correctness/```: 1 if the model prediction is correct, 0 otherwise. For example

```buildoutcfg
import pickle as pkl
qqp_size2correctness, qqp_data = pkl.load(open('correctness/qqp.pkl', 'rb')) # qqp_data is exactly data['predict'] as mentioned above
print(qqp_size2correctness.keys()) # output: dict_keys(['mini', 'small', 'medium', 'base', 'large']). qqp_size2correctness is a mapping from model size to the correctness tensor
print(qqp_size2correctness['large'].shape) # output: (79497, 10, 5, 4). 79497 is the number of datapoints, 10/5 is the number of pretraining/finetuning seeds, 4 represents different checkpoints at [3, 3.33, 3.67, 4] epochs. 
```

```correctness_p/```: Same as "correctness", the probability assigned to the correct class.

```ensemble_c/```: the correctness tensor using the last checkpoint only, after marginalizing over all fine-tuning seeds.

### 3. Code

Run ```pip3 install -r requirements.txt``` to install the required dependencies, and download the files to the corresponding folders.
The ```results.ipynb``` jupyter notebook computes the following:

#### 3.1 Variations within a Single Model Size

- within/across pretraining seed difference
- 0/1 loss bias-finetuningvariance-pretrainingvariance-decomposition
- variance conditioned on bias
- squared loss decomposition using probability assigned to the correct label

#### 3.2 Estimating the Fraction of Decaying Instances with Random Baseline

- Our approach, with CDF visualization
- Comparison with the conventional Benjamini-Hochberg Procedure
- Example decaying instances

The decaying instances are stored in the ```decaying/``` folder. Each .pkl file is a map from the data split name to a 3-tuple, which are

- a list of datapoints
- whether the datapoint belongs to the control group (random non-decaying instances), or the treatment group (decaying instances)
- Ruiqi Zhong's annotation of whether he thinks the label is correct, wrong, reasonable or unsure. (None means not annotated)

#### 3.3 Correlation of Instance Difference

- instance difference as measured by pearson-r correlation (numbers represent models sizes, smaller numbers correspond to smaller models)
- measuring with spearman rank

**Some results might be different from that in the paper due to random seeds; however they should be close and lead to the exact qualitative conclusions.**











