import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_var_cor(df: pd.DataFrame):
    corr = df.corr()
    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return corr


def matrix_distance_abs(ma, mb):
    return np.sum(np.abs(np.subtract(ma, mb)))


def matrix_distance_euclidian(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))


def wasserstein_distance(y, y_hat):
    return stats.wasserstein_distance(y, y_hat)


def get_duplicates(real_data, synthetic_data):
    df = pd.merge(real_data, synthetic_data.set_index('trans_amount'), indicator=True, how='outer')
    duplicates = df[df._merge == 'both']
    return len(duplicates), duplicates
