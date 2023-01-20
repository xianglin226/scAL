# scAL
An active learning approach for clustering single-cell RNA-seq data

![model](https://github.com/xianglin226/scAL/blob/master/src/AL_structure.jpg?raw=true)
# Example  
python Run_AL.py --data ./Datasets/10X_PBMC_select_2100_top2000.h5 --sn 50 --k 20 --budget 800 --split 0.7 --model SVM --method E --seed 1026

# Arguments
--data: the scRNA-seq data count matrix.  
--sn: size of the initial training set.  
--k: added cells in each active learning iteration.  
--budget: total cells that can be labeled by oracle.  
--split: train/test split ratio.  
--model: classifer used in active learning model. Options: 1) SVM; 2) LR (logistic regression); 3) RF (random forest); 4) MLP (multilayer proception).  
--method: sammple seletion algorithm. Options: E: entropy; M: margin; L: likelihood.  
--seed: randomness  

# Cite this work  
Lin, X., Liu, H., Wei, Z., Roy, S. B., & Gao, N. (2022). An active learning approach for clustering single-cell RNA-seq data. Laboratory Investigation, 102(3), 227-235.

