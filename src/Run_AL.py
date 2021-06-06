from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LogisticRegression as lr
from sklearn.svm import SVC as svm
from sklearn.neural_network import MLPClassifier as mlp
from sklearn import metrics
import os
import h5py
import numpy as np
import pandas as pd
from scipy.stats import entropy
import scanpy as sc
from preprocess import read_dataset, normalize
from ActiveLearning import Activelearning
from time import time
import random

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='AL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='10X_PBMC_select_2100_top2000.h5', help= 'data input')
    parser.add_argument('--sn', default=50, help='Initial training set')
    parser.add_argument('--k', default=20, help='Added cells in each iteration')
    parser.add_argument('--budget', default=500, help='Total cells from oracle')
    parser.add_argument('--split', default=0.3, help='Train test split')
    parser.add_argument('--model', default='SVM', help='Classifier')
    parser.add_argument('--method', default='E', help='Sample selection algorithm: E - entropy; M: margin; L: likelihood')
    parser.add_argument('--seed', default=1026, help='randomness')
    args = parser.parse_args()
    random.seed(args.seed)
    #Read data
    data_mat = h5py.File(args.data)
    x=np.array(data_mat["X"])
    y=np.array(data_mat["Y"])
    data_mat.close()
    
    #Normalization
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = read_dataset(adata,transpose=False,test_split=False,copy=True)
    adata = normalize(adata,size_factors=True,normalize_input=True,logtrans_input=True)
    
    #Get AL class
    AL = Activelearning(adata.X, y, k=args.k, sn=args.sn, budget=args.budget, split=args.split, model=args.model, method=args.method)
    
    #Run BL model
    out1 = AL.runBaseline()
    print("Baseline model:")
    print(out1)
    
    #Run AL model
    t0 = time()
    out2 = AL.runAL(verbose=False)
    print("Active model final performance: ")
    print(out2)
    print('Model:' + args.model)
    print('Total time: %d seconds.' % int(time() - t0))
