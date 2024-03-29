from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LogisticRegression
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

#AL model
class Activelearning():
    def __init__(self, x, y, k=20, sn=50, budget=500, split=0.3, model = "SVM",
                     method="M"):
        super(Activelearning, self).__init__()
        self.x = x
        self.y = y
        self.k = k
        self.sn = sn
        self.budget = budget
        self.split = split
        self.model = model
        self.method = method
        self.n = np.unique(y).shape[0]
#split data to pool and test
    def datapreprocess1(self, x, y, split, stratify):
        poolx, testx, pooly, testy = train_test_split(x, y, test_size=split, stratify=stratify)
        return poolx, testx, pooly, testy
    
    def setSNcells(self,x,y,sn, stratify):
        # make sure each class has been sampled once.
        tmp_x = []
        tmp_y = []
        shuffler = np.random.permutation(len(y))
        x = x[shuffler]
        y = y[shuffler]
        for idx in (set(y)) :
            tmp_x_1 = x[y==idx]
            tmp_x_2 = x[y!=idx]
            tmp_y_1 = y[y==idx]
            tmp_y_2 = y[y!=idx]
            last_x, tmp_x_1 = tmp_x_1[-1], tmp_x_1[:-1]
            last_y, tmp_y_1 = tmp_y_1[-1], tmp_y_1[:-1]
            tmp_x.append(last_x)
            tmp_y.append(last_y)
            x = np.concatenate((tmp_x_1, tmp_x_2), axis=0)
            y = np.concatenate((tmp_y_1, tmp_y_2), axis=0)
        
        tmp_x = np.array(tmp_x)
        tmp_y = np.array(tmp_y)
        #initialize the training set
        trainx,valix,trainy,valiy = train_test_split(x, y, train_size = sn, stratify=stratify)
        trainx = np.concatenate((tmp_x, trainx), axis=0)
        trainy = np.concatenate((tmp_y, trainy), axis=0)
        return trainx, valix, trainy, valiy
    
    def getmodel(self,x,y,model):
        if model == "SVM":
            SVC = svm(probability=True)
            SVC.fit(x, y)
            return SVC
        elif model == "RF":
            RF = rf()
            RF.fit(x, y)
            return RF
        elif model == "LR":
            #LR = lr()
            #LR.fit(x,y)
            #return LR
            LR = LogisticRegression(max_iter=1000)
            LR_fit = LR.fit(x, y)
            return LR_fit
        elif model == "MLP":
            #MLP = mlp(hidden_layer_sizes=[256, 128, 64])
            #MLP = mlp(hidden_layer_sizes=(128, 64, 32), batch_size=50)
            MLP = mlp(hidden_layer_sizes=(256, 128, 64, 32, 16), batch_size=50)
            MLP.fit(x, y)
            return MLP
        else:
            print("Wrong model input!")
            raise ValueError
    
    def performance(self, pred, y, proba):
        acc = np.round(metrics.accuracy_score(pred, y), 5)
        auc = np.round(metrics.roc_auc_score(y, proba, multi_class="ovr"), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(pred, y), 5)
        ari = np.round(metrics.adjusted_rand_score(pred, y), 5)
        recall = np.round(metrics.recall_score(pred, y, average= "weighted"), 5)
        precision = np.round(metrics.precision_score(pred, y, average= "weighted"), 5)
        f1 = np.round(metrics.f1_score(pred, y, average= "weighted"), 5)
        return {"acc":acc,"auc":auc,"nmi":nmi,"ari":ari,"recall":recall,"precision":precision,"F1":f1}
    
    def sampleSelection(self,p,n,k,method):
    #p is the probability matrix
    #n is the total number of clusters
    #k is the number of selected cells
    #method is the algorithm used for cell selection
    #return the index of the selected cells
        if method == "M":
            mar = []
            for i in range(p.shape[0]):
                m = p.argsort()
                sec = p[i][np.where(m[i]==n-2)]
                fir = p[i][np.where(m[i]==n-1)]
                margin = fir - sec
                mar.append(margin)
            add = np.array(mar).flatten()
            return add.argsort()[:k]
        
        elif method == "L":
            mar = []
            for i in range(p.shape[0]):
                m = p.argsort()
                fir = p[i][np.where(m[i]==n-1)]
                mar.append(fir)
            add = np.array(mar).flatten()
            return add.argsort()[:k]
        
        elif method == "E":
            ent = []
            for i in range(p.shape[0]):
                m = p.argsort()
                en = entropy(p[i])
                ent.append(en)
            add = np.array(ent).flatten()
            return add.argsort()[p.shape[0]-k:]
        
        else:
            print("Wrong method input!")
            raise ValueError
    
    def train_val_update(self, trainx, trainy, valx, valy, add):
        add_cell, add_label = valx[add], valy[add]
        trainx_, trainy_ = np.concatenate([trainx,add_cell]), np.concatenate([trainy,add_label])
        valx_,valy_ = np.delete(valx, add, 0), np.delete(valy, add, 0)
        return trainx_, trainy_, valx_, valy_
    
    def runBaseline(self):
        poolx, testx, pooly, testy = self.datapreprocess1(self.x, self.y, self.split, stratify=self.y)
        blx, _, bly, _ = self.setSNcells(poolx, pooly, self.budget, stratify=None)
        clf = self.getmodel(blx,bly,self.model)
        pred = clf.predict(testx)
        proba = clf.predict_proba(testx)
        return self.performance(pred, testy, proba)
    
    def runAL(self, verbose):
        poolx, testx, pooly, testy = self.datapreprocess1(self.x, self.y, self.split, stratify=self.y)
        snx,valx,sny,valy = self.setSNcells(poolx, pooly, self.sn, stratify=None)
        i = 0
        while snx.shape[0] < self.budget:
            clf = self.getmodel(snx,sny,self.model)
            prob = clf.predict_proba(valx)
            add = self.sampleSelection(p=prob, n=self.n, k=self.k, method=self.method)
            snx,sny,valx,valy = self.train_val_update(snx,sny,valx,valy,add)
            if verbose == True and i%5 == 0:
               pred = clf.predict(testx)
               proba = clf.predict_proba(testx)
               print('Iteration: ' + str(i) + ', training size: ', str(snx.shape[0]))
               print(self.performance(pred, testy, proba))
            i += 1
        
        #predict on test set or on whole dataset.
        pred = clf.predict(testx)
        proba = clf.predict_proba(testx)
        return self.performance(pred, testy, proba)
        #pred = clf.predict(self.x)
        #proba = clf.predict_proba(self.x)
        #return self.performance(pred, self.y, proba)
