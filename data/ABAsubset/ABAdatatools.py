"""
Functions of load and parse ABA ISH data

Last Update: Novemver 2022
"""

import pandas as pd
from pathlib import Path


def load_counts():
    """
    read in raw counds table
    Args: None
    Returns:
        df - pandas dataframe containing all counts data
    """

    # path in repo
    filepath = Path("./data/ABAsubset/ABAISHsubset.csv")

    # load data
    df = pd.read_csv(filepath, index_col=0)

    return df


def load_meta():
    """
    read in meta data table
    Args: None
    Returns:
        metadf - pandas dataframe containing all meta data for the counts table
    """

    # path in repo
    filepath = Path("./data/ABAsubset/ABAISHsubset_labels.csv")

    # load meta data
    metadf = pd.read_csv(filepath, index_col=0, header=None)
    metadf.columns = ["thalamus"]

    return metadf
