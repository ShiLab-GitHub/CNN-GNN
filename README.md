This reposiry contains the data and python scripts in support of the ICBBE'2024 paper: [A deep learning model of BACE-1 inhibitors to reveal molecular interactions using graph neural networks and convolutional neural networks](https://doi.org/10.1117/12.3044459 ).

# Resources:

+ README.md: this file.
+ bace.csv  
  this file was adapted from https://github.com/ahrbadr/QSAR-Model-for-BACE1/tree/main

###  Source codes:
+ create_data.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
+ training.py: train a CNN-GNN model.
+ model/cnn-gnn.py: proposed model CNN-GNNNet receiving graphs and string as input for drugs.

# Step-by-step running:

## 1. Install Python libraries needed
+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
+ Install rdkit: conda install -y -c conda-forge rdkit
+ Install sklearn: pip install scikit-learn
+ Install pandas: pip install pandas


## 2. Create data in pytorch format
Running
```sh
conda activate geometric
python create_data.py
```
This returns bace_train.csv, bace_test.csv saved in data. These files are in turn input to create data in pytorch format,
stored at data/processed/, consisting of  bace_train.pt, bace_test.pt.

## 3. Train the prediction model
To train the model using training data.   
Running 

```sh
conda activate geometric
python training.py
```

Then you will get the RMSE value and IC value for each epoch and the outcome curve of ACC and AUC for the classification task of BACE1 inhibitors activity.


