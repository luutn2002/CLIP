import pandas as pd
import numpy as np
from scipy import stats

def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    srocc_result = plcc(xranks, yranks)
    return srocc_result

def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    x, y = np.float32(x), np.float32(y)
    plcc_result = stats.pearsonr(x, y)[0]
    return np.round(plcc_result,3)