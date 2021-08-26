# Author: Lalitha Viswanathan
# Project: rna seq data analysis using numpy / scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def readsperkilobasepermilliontranscripts(counts, lengths):
    N = np.sum(counts, axis=0)  # sum each column to get total reads per sample
    L = lengths
    C = counts

    normed = 1e9 * C / (N[np.newaxis, :] * L[:, np.newaxis])

    return(normed)

def binned_boxplot(x, y):
    x_hist, x_bins = np.histogram(x)
    x_bin_idxs = np.digitize(x, x_bins[:-1])
    binned_y = [
        y[x_bin_idxs == i]
                for i in range(np.max(x_bin_idxs))]
    fig, ax = plt.subplots(figsize=(4.8, 1))
    ax.boxplot(binned_y)
#def reduce_xaxis_labels(Ax, factor):




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    data_table = pd.read_csv(r"C:\Users\visu4\Documents\wcgna\data\counts.csv", index_col=0)
    print(data_table.iloc[:5, :5])
    print("Genes in data_table: ", data_table.shape[0])
    print("Index for data table")
    print(data_table.index)
    gene_info = pd.read_csv(r"C:\Users\visu4\Documents\wcgna\data\genes.csv", index_col=0)
    print(gene_info.iloc[:5, :5])
    matched_index = pd.Index.intersection(data_table.index, gene_info.index)
    print(matched_index)
    print(matched_index.shape[0]);
    counts = np.asarray(data_table.loc[matched_index], dtype=int)
    gene_names = np.array(matched_index)
    print(f'{counts.shape[0]} genes measured in {counts.shape[1]} individuals.')
    gene_lengths = np.asarray(gene_info.loc[matched_index]['GeneLength'],
                              dtype=int)
    print(counts.shape)
    print(gene_lengths.shape)

    total_counts = np.sum(counts, axis=0)  # sum columns together
    density = stats.kde.gaussian_kde(total_counts)
    x = np.arange(min(total_counts), max(total_counts), 10000)
    # Make the density plot
    fig, ax = plt.subplots()
    ax.plot(x, density(x))
    ax.set_xlabel("Total counts per individual")
    ax.set_ylabel("Density")

    plt.show()
    print(f'Count statistics:\n  min:  {np.min(total_counts)}'
          f'\n  mean: {np.mean(total_counts)}'
          f'\n  max:  {np.max(total_counts)}')

    np.random.seed(seed=7)  # Set seed so we will get consistent results
    # Randomly select 70 samples
    samples_index = np.random.choice(range(counts.shape[1]), size=70, replace=False)
    counts_subset = counts[:, samples_index]

    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.boxplot(counts_subset)
    ax.set_xlabel("Individuals")
    ax.set_ylabel("Gene expression counts")
    # Box plot of expression values before normalization
    ax.boxplot(np.log(counts_subset+1))
    plt.show()

    # Plot of normalized counts
    counts_lib_norm = counts / total_counts * 1000000
    counts_subset_lib_norm = counts_lib_norm[:, samples_index]
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.boxplot(np.log(counts_subset_lib_norm + 1))
    plt.show()


    #RPKM
    C = counts
    N = counts.sum(axis=0)
    L = gene_lengths
    L = L[:, np.newaxis]
    C_tmp = (10^9 * C) / L
    rpkm_counts = C_tmp / N[np.newaxis, :]

    counts_rpkm = readsperkilobasepermilliontranscripts(counts, gene_lengths)
    log_counts = np.log(counts_rpkm+1)
    mean_log_counts = np.mean(log_counts, axis=1)
    log_gene_lengths = np.log(gene_lengths)

    binned_boxplot(x=log_gene_lengths, y=mean_log_counts)
