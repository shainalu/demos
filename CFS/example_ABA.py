"""
Example running CFS on ABA ISH brain data

Last Modified: November 2022
"""

# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# imports from this repo
from data.ABAsubset.ABAdatatools import load_counts, load_meta
from CFS.cfs import zscore, applyCFS

# example data: ABA brain
df = load_counts()
metadf = load_meta()

# train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    df,
    metadf.thalamus,
    test_size=0.2,
    random_state=9,
    shuffle=True,
    stratify=metadf.thalamus,
)

# z-score train and test folds
Xtrain = zscore(Xtrain)
Xtest = zscore(Xtest)

# CFS
featset, train_auroc, test_auroc = applyCFS(Xtrain, Xtest, ytrain, ytest, 10, 10)

# outputs
summary = pd.concat([featset, train_auroc, test_auroc], axis=1)
summary.columns = ["feature sets", "train AUROC", "test AUROC"]
print(summary)
summary.to_csv("featuresets.csv")
