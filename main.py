# Author: Lalitha Viswanathan
# Project: rna seq data analysis using numpy / scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

# Deseq normalization
def size_factors(counts):
    counts = counts[np.alltrue(counts)]
    logcounts = np.log(counts)
    loggeommeans = np.mean(logcounts, axis=1).reshape(len(logcounts), 1)
    # deseq normalization is exp of median of log counts - log of geometric mean of counts
    sf = np.exp(np.median(logcounts - loggeommeans, axis=0))
    return sf

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
    print(matched_index.shape[0])
    counts = np.asarray(data_table.loc[matched_index], dtype=int)
    gene_names = np.array(matched_index)
    print(f'{counts.shape[0]} genes measured in {counts.shape[1]} individuals.')
    gene_lengths = np.asarray(gene_info.loc[matched_index]['GeneLength'],
                              dtype=int)
    print(counts.shape)
    print(gene_lengths.shape)

    total_counts = np.sum(counts, axis=0)  # sum columns together
    density = stats.kde.gaussian_kde(total_counts)
    #x = np.arange(min(total_counts), max(total_counts), 10000)
    # Make the density plot
    #fig, ax = plt.subplots()
    #ax.plot(x, density(x))
    #ax.set_xlabel("Total counts per individual")
    #ax.set_ylabel("Density")

    #plt.show()
    print(f'Count statistics:\n  min:  {np.min(total_counts)}'
          f'\n  mean: {np.mean(total_counts)}'
          f'\n  max:  {np.max(total_counts)}')

    #np.random.seed(seed=7)  # Set seed so we will get consistent results
    # Randomly select 70 samples
    #samples_index = np.random.choice(range(counts.shape[1]), size=70, replace=False)
    #counts_subset = counts[:, samples_index]

    #fig, ax = plt.subplots(figsize=(4.8, 2.4))
    #ax.boxplot(counts_subset)
    #ax.set_xlabel("Individuals")
    #ax.set_ylabel("Gene expression counts")
    # Box plot of expression values before normalization
    #ax.boxplot(np.log(counts_subset+1))
    #plt.show()

    # Plot of normalized counts
    counts_lib_norm = counts / total_counts * 1000000
    #counts_subset_lib_norm = counts_lib_norm[:, samples_index]
    #fig, ax = plt.subplots(figsize=(4.8, 2.4))
    #ax.boxplot(np.log(counts_subset_lib_norm + 1))
    #plt.show()

    # Normalize counts using rpkm
    counts_rpkm = readsperkilobasepermilliontranscripts(counts, gene_lengths)
    print(counts_rpkm)
    np.seterr(invalid='ignore')
    log_counts = np.log(counts_rpkm + 1)
    mean_log_counts = np.mean(log_counts, axis=1)
    log_gene_lengths = np.log(gene_lengths)

    print(log_counts)
    print(mean_log_counts)
    print(log_gene_lengths)


 #######################################################
    plt.hist(total_counts)
    plt.show()
    (k2, pval) = stats.mstats.normaltest(total_counts)
    print(k2)
    print(pval)

    # Min Max rescaling for gene expression counts

    total_counts_scaled = np.array([(x - np.min(total_counts)) / (np.max(total_counts) - np.min(total_counts)) for x in total_counts])
    print(total_counts_scaled)

    #total_counts = total_counts.reshape(-1, 1)
    #print(preprocessing.normalize(total_counts))

    print("Deseq normalization")
    print(size_factors(total_counts))

    print("Np lin alg normalization")
    print (len(total_counts))
    #total_counts_reshaped = total_counts.reshape(15,25)
    print(total_counts / np.linalg.norm(total_counts))


    arr1 = np.arange(12)
    print(arr1)
    print(np.linalg.norm(arr1))
    ##########################################################

    #np.histogram(log_gene_lengths)
    #plt.hist(log_gene_lengths, bins='auto')
    #plt.show()

    (k2, pval) = stats.mstats.normaltest(log_gene_lengths)
    print(k2)
    print(pval)

    ##########################################################

    #binned_boxplot(x=log_gene_lengths, y=mean_log_counts)

    #RPKM
    #C = counts
    #N = counts.sum(axis=0)
    #L = gene_lengths
    #L = L[:, np.newaxis]
    #C_tmp = (10^9 * C) / L
    #rpkm_counts = C_tmp / N[np.newaxis, :]

