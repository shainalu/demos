# imports
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from sklearn.preprocessing import StandardScaler
import random

######## transformations ########
def zscore(df):
    """
    z-score data 
    Args: 
        df - pandas dataframe where rows are samples and columns are features
    Returns:
        z_df - pandas dataframe where input dataframe is z-scored; rows are samples and columns are features
    """

    # z-score
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(df)
    z_df = scaler.transform(df)

    # store z-scored df as pandas dataframe
    z_df = pd.DataFrame(z_df)
    z_df.columns = df.columns
    z_df.index = df.index

    return z_df


######## AUROC ########
def analytical_auroc(featurevector, binarylabels):
    """
    analytical calculation of auroc
    Args: 
        featurevector - vector to rank by
        binarylabels - vector of 1's and 0's indicating class membership
    Returns: 
        auroc - area under the receiver operating curve
    """

    # sort feature and binary labels by feature vectors, ascending
    s = sorted(zip(featurevector, binarylabels))
    feature_sort, binarylabels_sort = map(list, zip(*s))

    # get the sum of the ranks in feature vector corresponding to 1's in binary vector
    sumranks = 0
    for i in range(len(binarylabels_sort)):
        if binarylabels_sort[i] == 1:
            sumranks = sumranks + feature_sort[i]

    poslabels = binarylabels.sum()
    neglabels = len(binarylabels) - poslabels

    auroc = (sumranks / (neglabels * poslabels)) - ((poslabels + 1) / (2 * neglabels))

    return auroc


######## CFS functions ########
def calcDE(Xtrain, ytrain):
    """
    calculate differential expression for all genes/features between the 2 classes
    Args:
        Xtrain - pandas dataframe, training fold of dataset of interest
        ytrain - pandas series, target variable for training fold of above dataset
    Returns:
        allpvals - pandas dataframe, p-values for both directions of MWU for all features
                   across the 2 classes of ytrain
    """

    # Ha: areaofinterest > not areaofinterest; i.e. alternative = greater
    Xtrain_ranked = Xtrain.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)
    n1 = ytrain.sum()  # instances of target var marked as 1
    n2 = len(ytrain) - n1
    U = Xtrain_ranked.loc[ytrain == 1, :].sum() - ((n1 * (n1 + 1)) / 2)

    T = Xtrain_ranked.apply(lambda col: sp.stats.tiecorrect(col), axis=0)
    sd = np.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)
    meanrank = n1 * n2 / 2.0 + 0.5
    z = (U - meanrank) / sd

    pvals_greater = pd.Series(
        stats.norm.sf(z), index=list(Xtrain), name="pvals_greater"
    )

    # Ha: areaofinterest < notareaofinterest; i.e. alternative = less
    Xtrain_ranked = Xtrain.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)
    n2 = ytrain.sum()  # instances of target var marked as 1
    n1 = len(ytrain) - n1
    U = Xtrain_ranked.loc[ytrain == 0, :].sum() - ((n1 * (n1 + 1)) / 2)

    T = Xtrain_ranked.apply(lambda col: sp.stats.tiecorrect(col), axis=0)
    sd = np.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)
    meanrank = n1 * n2 / 2.0 + 0.5
    z = (U - meanrank) / sd

    pvals_less = pd.Series(stats.norm.sf(z), index=list(Xtrain), name="pvals_less")

    allpvals = pd.concat([pvals_greater, pvals_less], axis=1)

    return allpvals


def getDEgenes(allpvals, numtotal):
    """
    helper function to return top most significantly different genes
    Args:
        allpvals - pandas dataframe, output from calcDE()
        numtotal - int, how many tops genes to return
    Returns:
        topDEgenes - pandas dataframe, with the top numtotal DE genes form allpvals
    """

    # melt
    allpvals["gene"] = allpvals.index
    allpvals_melt = allpvals.melt(id_vars="gene")
    # sort by p-value
    allpvals_melt = allpvals_melt.sort_values(by="value", ascending=True)
    # get top X number of DE genes
    topDEgenes = allpvals_melt.iloc[0:numtotal, :]

    return topDEgenes


def get_featset(featurecorrs, ranksdf, ylabels, seedgene):
    """
    implementation of correlation-based feature selection using forward selection
    stops when average auroc prediciton is no longer improving
    picks subsequent genes that are the lowest correlated to current genes on average, first of ties
    stop when average auroc prediction is no longer improving
    Args: 
        featurecorrs - correlation matrix of features being considered
        ranksdf - pandas dataframe, top DE genes transformed to ranks across samples
        ylabels - pandas series, target variable 
        seedgene - string, feature name to start CFS, should be randomly selected
    Returns: 
        featset - found CFS feature set
    """

    # start with passed in randomly picked gene
    featset = [seedgene]
    # get starting performance
    curr_auroc = analytical_auroc(
        sp.stats.mstats.rankdata(ranksdf.loc[:, featset].mean(axis=1)), ylabels
    )
    improving = True
    while improving:
        # look at all other possible features and take lowest correlated to seed, others in feat set
        means = featurecorrs.loc[:, featset].mean(
            axis=1
        )  # get average corr across choosen features
        featset.append(
            means.idxmin()
        )  # gets row name of min mean corrs, picks first of ties
        # check featset performance
        new_auroc = analytical_auroc(
            sp.stats.mstats.rankdata(ranksdf.loc[:, featset].mean(axis=1)), ylabels
        )

        if new_auroc <= curr_auroc:  # if not improved, stop
            featset.pop(len(featset) - 1)
            final_auroc = curr_auroc
            improving = False
        else:
            curr_auroc = new_auroc

    return featset


def applyCFS(Xtrain, Xtest, ytrain, ytest, numtopDE, numsets):
    """
    run correlation-based feature selection on train set and return AUROC for train and test folds
    Args:
        Xtrain - pandas dataframe, train fold samples (rows) by features (cols)
        Xtest - pandas dataframe, test fold samples (rows) by features (cols)
        ytrain - pandas series, target variable for Xtrain
        ytest - pandas series, target varible for Xtest
        numtopDE - int, how many top DE genes to take
        numsets - int, how many CFS sets to find
    Returns:
        featsets - pandas series containing lists of all feature sets
    """

    # calculate DE genes across the two brain areas
    allpvals = calcDE(Xtrain, ytrain)
    # get top X DE genes
    topDEgenes = getDEgenes(allpvals, numtopDE)

    # ranks DE genes
    # train
    rankedXtrain = Xtrain.loc[:, topDEgenes.gene]
    rankedXtrain.loc[:, (topDEgenes.variable == "pvals_less").values] = (
        -1 * rankedXtrain.loc[:, (topDEgenes.variable == "pvals_less").values]
    )
    rankedXtrain = rankedXtrain.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)
    # test
    rankedXtest = Xtest.loc[:, topDEgenes.gene]
    rankedXtest.loc[:, (topDEgenes.variable == "pvals_less").values] = (
        -1 * rankedXtest.loc[:, (topDEgenes.variable == "pvals_less").values]
    )
    rankedXtest = rankedXtest.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)

    # correlation matrix (spearman, b/c already ranked)
    traincorrs = np.corrcoef(rankedXtrain.values.T)
    traincorrs = pd.DataFrame(
        traincorrs, index=topDEgenes.gene.values, columns=topDEgenes.gene.values
    )

    # get feature sets using CFS
    random.seed(42)
    startingpts = pd.Series(random.sample(list(traincorrs), numsets))
    featsets = startingpts.apply(
        lambda x: get_featset(traincorrs, rankedXtrain, ytrain, x)
    )
    trainaurocs = featsets.apply(
        lambda x: analytical_auroc(
            sp.stats.mstats.rankdata(rankedXtrain.loc[:, x].mean(axis=1)), ytrain
        )
    )
    testaurocs = featsets.apply(
        lambda x: analytical_auroc(
            sp.stats.mstats.rankdata(rankedXtest.loc[:, x].mean(axis=1)), ytest
        )
    )

    # return all 100 feature sets and aurocs
    return featsets, trainaurocs, testaurocs
