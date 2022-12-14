"""
Example running CFS on sklearn breast cancer data

Last Modified: November 2022
"""

# imports
import pandas as pd
from sklearn.model_selection import train_test_split

# imports from this repo
from CFS.cfs import zscore, applyCFS

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data["data"])
target = pd.Series(data["target"])

# train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    df, target, test_size=0.2, random_state=9, shuffle=True, stratify=target
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
# summary.to_csv("featuresets.csv")
