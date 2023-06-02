# ReLMole
Supplement of code for "ReLMole: Molecular Representation Learning based on Two-Level Graph Similarities"

## Requirements
We implement our model on `Python 3.7`. These packages are mainly used:
```
torch                     1.7.1
torch-cluster             1.5.8              
torch-geometric           1.6.3
torch-scatter             2.0.5 
torch-sparse              0.6.8
torch-spline-conv         1.2.0
numpy                     1.19.5
scikit-learn              0.24.1
rdkit                     2020.09.1.0
deepchem                  2.5.0
```

## Dataset
#### 1. Pre-training dataset
We use 250k "lead-like" compounds from [ZINC15](https://zinc15.docking.org), which is available in DeepChem package. You get get it by calling `deepchem.molnet.load_zinc15()` function. We convert the dataset into text format and saved it into `data/ZINC15/zinc15_250k`.

#### 2. Molecular property datasets
We load MoleculeNet datasets using DeepChem package.

#### 3. DDI datasets
We download the DDI datasets from [CASTER](https://github.com/kexinhuang12345/CASTER). The splitting results used in ReLMole are available in directory `data/DDI`


## Experiments
#### 1. FG corpus generating
Run `gen_fg_corpus.py` to generate FG corpus and the corpus file will be saved into `data/ZINC15/fg_corpus.txt`.

#### 2. CDF of similarities of two-level graphs
Run `sim_cdf.py` to sample molecule pairs from the pre-training dataset and plot the CDF curve of two-level similarities. The figures will be saved in directory `data/ZINC15`.

#### 3. Pre-training
Run `pretrain_cl.py` to pre-train ReLMole and the pre-trained model will be saved in directory `pretrained_model_cl_zinc15_250k`.

#### 4. Fine-tuning
For the molecular property prediction task, run `task_property/run_${dataset}.py` to fine-tune the pre-trained model. \
For the DDI prediction task, run `task_ddi/run_ddi.py` to fine-tune the pre-trained model. \
We apply fine-tuned models for each dataset in directory `finetuned`.