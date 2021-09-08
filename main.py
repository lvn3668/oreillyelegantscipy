# Author: Lalitha Viswanathan
# Project: rna seq data analysis using numpy / scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram, leaves_list, fcluster
import itertools as it
from collections import defaultdict

##########################################################
def clear_spines(axes):
    for loc in ['left', 'right', 'top', 'bottom']:
        axes.spines[loc].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])

##########################################################

def plot_bicluster(data, row_linkage, col_linkage, row_nclusters=10, col_nclusters=3):
    fig = plt.figure(figsize=(4.8, 4.8))
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    threshold_r = (row_linkage[-row_nclusters, 2] + row_linkage[-row_nclusters + 1, 2]) / 2
    with plt.rc_context({'lines.linewidth': 0.75}):
        dendrogram(row_linkage, orientation='left', color_threshold=threshold_r, ax=ax1)
    clear_spines(ax1)
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    threshold_c = (col_linkage[-col_nclusters, 2] + col_linkage[-col_nclusters + 1, 2]) / 2
    with plt.rc_context({'lines.linewidth': 0.75}):
        dendrogram(col_linkage, orientation='left', color_threshold=threshold_c, ax=ax2)
    clear_spines(ax2)
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx_rows = leaves_list(row_linkage)
    data = data[idx_rows, :]
    print(len(data))
    idx_cols = leaves_list(col_linkage)
    print(idx_cols)
    #data = data[:, idx_cols]
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='YlGnBu_r')
    clear_spines(ax)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Genes', labelpad=125)
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)
    plt.show()

##########################################################
def survival_distribution_function(lifetimes, right_censored=None):
    n_obs = len(lifetimes)
    rc = np.isnan(lifetimes)
    if right_censored is not None:
        rc|=right_censored
    observed = lifetimes[-rc]
    xs = np.concatenate(([0], np.sort(observed)))
    ys = np.linspace(1,0,n_obs+1)
    ys = ys[:len(xs)]
    return xs, ys
##########################################################
def plot_cluster_survival_curves(clusters, sample_names, patients, censor=True):
    fig, ax = plt.subplots()
    if type(clusters) == np.ndarray:
        cluster_ids = np.unique(clusters)
        cluster_names = ['cluster {}'.format(i) for i in cluster_ids ]
    elif type(clusters) == pd.Series:
        cluster_ids = clusters.cat.categories
        cluster_names = list(cluster_ids)
    n_clusters = len(cluster_ids)
    #print(n_clusters)
    #print(sample_names)
    for c in cluster_ids:
        clust_samples = np.flatnonzero(clusters == c)
        print(patients.index)
        print(sample_names)
        print("Inside for loop")
        #sample_names = sample_names[:333]
        print(len(sample_names))
        #for i in clust_samples[:len(sample_names)-2]:
        #i=0
        clust_samples = [sample_names[i] for i in clust_samples if sample_names[i] in patients.index]
        #while (i < len(sample_names)):
        print("Inside inner for loop")
        #print(i)
        #print(patients.index[i])
        #if (sample_names[i] in patients.index):
                # Should throw error here
                #print(i)
                #print(patients.index[i])
                #print(sample_names[i])
        #clust_samples = sample_names[i]
        #clust_samples = [sample_names[i] for i in clust_samples if sample_names[i] in patients.index]
        patient_cluster = patients.loc[clust_samples]
        survival_times = patient_cluster['melanoma-survival-time'].values
        print("Before clustering on dead melanoma patients")
        if censor:
            censored = -patient_cluster['melanoma-dead'].values.astype(bool)
        else:
            censored = None
        print("Survival Distribution Function")
        stimes, sfracs = survival_distribution_function(survival_times, censored)
        ax.plot(stimes/365, sfracs)

    print("Before Plot.show")
    plt.show()
##########################################################
# Deseq normalization
def size_factors(counts):
    counts = counts[np.alltrue(counts)]
    logcounts = np.log(counts)
    loggeommeans = np.mean(logcounts, axis=1).reshape(len(logcounts), 1)
    # deseq normalization is exp of median of log counts - log of geometric mean of counts
    sf = np.exp(np.median(logcounts - loggeommeans, axis=0))
    return sf

##########################################################
def rpkm(counts, lengths):
    N = np.sum(counts, axis=0)  # sum each column to get total reads per sample
    L = lengths
    C = counts

    normed = 1e9 * C / (N[np.newaxis, :] * L[:, np.newaxis])

    return (normed)

##########################################################
def binned_boxplot(x, y, *, xlabel='gene length(log scale)', ylabel='average log counts'):
    x_hist, x_bins = np.histogram(x)
    x_bin_idxs = np.digitize(x, x_bins[:-1])
    binned_y = [
        y[x_bin_idxs == i]
        for i in range(np.max(x_bin_idxs))]
    fig, ax = plt.subplots(figsize=(4.8, 1))
    plt.title("Binned box plot of gene length(log scale) v/s average log counts")
    x_bin_centers = (x_bins[1:] + x_bins[:-1])/2
    x_ticklabels = np.round(np.exp(x_bin_centers)).astype(int)
    ax.boxplot(binned_y, labels=x_ticklabels)
    reduce_xaxis_labels(ax, 10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

##########################################################
def reduce_xaxis_labels(ax, factor):
    plt.setp(ax.xaxis.get_ticklabels(), visible = False)
    for label in ax.xaxis.get_ticklabels()[factor-1::factor]:
        label.setvisible(True)


##########################################################
def quantile_norm(x):
    quantiles = np.mean(np.sort(x, axis=0), axis=1)
    ranks = np.apply_along_axis(stats.rankdata, 0, x)
    rank_indices = ranks.astype(int) - 1
    xn = quantiles[rank_indices]
    return (xn)


def quantile_norm_log(x):
    logx = np.log(x + 1)
    logxn = quantile_norm(logx)
    return logxn
##########################################################


def plot_col_density(data):
    density_per_col = [stats.gaussian_kde(col) for col in data.T]
    x = np.linspace(np.min(data), np.max(data), 100)
    flag, ax = plt.subplots()
    for density in density_per_col:
        ax.plot(x, density(x))
    plt.show()


##########################################################

def class_boxplot(data, classes, colors=None, **kwargs):
    all_classes = sorted(set(classes))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    class2color = dict(zip(all_classes, it.cycle(colors)))
    class2data = defaultdict(list)
    for distrib, cls in zip(data, classes):
        for c in all_classes:
            class2data[c].append([])
        class2data[cls][-1] = distrib

    fig, ax = plt.subplots()
    lines = []
    for cls in all_classes:
        for key in ['boxprops', 'whiskerprops', 'flierprops']:
            kwargs.setdefault(key, {}).update(color=class2color[cls])
        box = ax.boxplot(class2data[cls],**kwargs)
        lines.append(box['whiskers'][0])
    ax.legend(lines, all_classes)
    return ax

##########################################################

def most_variable_rows(data, *, n=375):
    rowvar = np.var(data, axis=1)
    sort_indices = np.argsort(rowvar)[-n:]
    variable_data = data[sort_indices, :]
    return variable_data


##########################################################

def bicluster(data, linkage_method='average', distance_metric='correlation'):
    y_cols = linkage(data, method=linkage_method, metric=distance_metric)
    x_rows = linkage(data.T, method=linkage_method, metric=distance_metric)
    return x_rows, y_cols

##########################################################
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_table = pd.read_csv(r"C:\Users\visu4\Documents\wcgna\data\counts.csv", index_col=0)
    print(data_table.iloc[:5, :5])
    samples = list(data_table.columns)
    print("Genes in data_table: ", data_table.shape[0])
    print("Index for data table")
    print(data_table.index)
    gene_info = pd.read_csv(r"C:\Users\visu4\Documents\wcgna\data\genes.csv", index_col=0)
    print(gene_info.iloc[:5, :5])
    print( "Genes in data table ",  data_table.shape[0])
    print("Genes in gene info", gene_info.shape[0])
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
    x = np.arange(min(total_counts), max(total_counts), 10000)
    # Make the density plot
    fig, ax = plt.subplots()
    ax.plot(x, density(x))
    ax.set_xlabel("Total counts per individual")
    ax.set_ylabel("Density")
    plt.title("Plot of total counts per individual v/s density")
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
    reduce_xaxis_labels(ax, 5)
    plt.title("Individuals v/s gene expression counts")
    plt.show()

    # log scale gene expression counts per individual
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.boxplot(np.log(counts_subset+1))
    ax.set_xlabel('Individuals')
    ax.set_ylabel("Log Gene expression counts")
    reduce_xaxis_labels(ax, 5)
    plt.title("Individuals v/s log gene expr counts")
    plt.show()

    # Plot of normalized counts where normalization is by library size
    counts_lib_norm = counts / total_counts * 1000000
    counts_subset_lib_norm = counts_lib_norm[:, samples_index]
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.boxplot(np.log(counts_subset_lib_norm + 1))
    plt.title("Plot of log gene expr counts normalized by lib size spread over individuals")
    ax.set_xlabel("Individuals")
    ax.set_ylabel("Log gene expression counts normalized by library size")
    reduce_xaxis_labels(ax, 5)
    plt.show()

    log_counts_3 = list(np.log(counts.T[:3] +1))
    log_ncounts_3 = list(np.log(counts_lib_norm.T[:3]+1))
    ax = class_boxplot(log_ncounts_3+log_counts_3, ['raw counts']*3 + ['normalized by library size']*3,
                       labels=[1,2,3,1,2,3])
    ax.set_xlabel('sample number')
    ax.set_ylabel('log gene expression counts')
    plt.title("Sample number v/s log gene expression counts")
    plt.show()

    print("Gene expression counts normalized by library size")
    log_of_gene_expr_counts_normalized_by_lib_size = np.log(counts_lib_norm+1)
    mean_log_counts = np.mean(log_of_gene_expr_counts_normalized_by_lib_size, axis=1)
    log_gene_lengths = np.log(gene_lengths)
    print(log_of_gene_expr_counts_normalized_by_lib_size)
    print("Mean log counts")
    print(mean_log_counts)
    print("Log gene lengths")
    print(log_gene_lengths)
    binned_boxplot(x=log_gene_lengths, y=mean_log_counts)

    # Normalize counts using rpkm
    counts_rpkm = rpkm(counts, gene_lengths)
    print("Gene expression counts normalizedd by reads per kilobase per million")
    print(counts_rpkm)
    np.seterr(invalid='ignore')
    log_counts = np.log(counts_rpkm + 1)
    mean_log_counts = np.mean(log_counts, axis=1)
    log_gene_lengths = np.log(gene_lengths)

    binned_boxplot(x=log_gene_lengths, y=mean_log_counts)
    #######################################################

    gene_idxs = np.array([80, 186])
    gene1, gene2 = gene_names[gene_idxs]
    len1, len2 = gene_lengths[gene_idxs]
    gene_labels = [f'{gene1}, {len1}bp', f'{gene2}, {len2}bp' ]
    log_counts = list(np.log(counts[gene_idxs]+1))
    log_ncounts = list(np.log(counts_rpkm[gene_idxs]+1))
    ax = class_boxplot(log_counts, ['raw counts']*3, labels=gene_labels)
    ax.set_xlabel('Genes')
    ax.set_ylabel('log gene expression counts over all samples')
    ax = class_boxplot(log_ncounts, ['RPKM normalized']*3, labels=gene_labels)
    ax.set_xlabel('Genes')
    ax.set_ylabel(' log RPKM gene expression counts over all samples')
    #plt.show()
    print(log_counts)
    print(mean_log_counts)
    print(log_gene_lengths)
    plt.show()
    exit()
    #######################################################
    # plt.hist(total_counts)
    # plt.show()
    (k2, pval) = stats.mstats.normaltest(total_counts)
    print(k2)
    print(pval)

    # Min Max rescaling for gene expression counts

    total_counts_scaled = np.array(
        [(x - np.min(total_counts)) / (np.max(total_counts) - np.min(total_counts)) for x in total_counts])
    print(total_counts_scaled)

    # total_counts = total_counts.reshape(-1, 1)
    # print(preprocessing.normalize(total_counts))

    print("Deseq normalization")
    print(size_factors(total_counts))

    print("Np lin alg normalization")
    print(len(total_counts))
    # total_counts_reshaped = total_counts.reshape(15,25)
    print(total_counts / np.linalg.norm(total_counts))
    arr1 = np.arange(12)
    print(arr1)
    print(np.linalg.norm(arr1))
    ##########################################################

    # np.histogram(log_gene_lengths)
    # plt.hist(log_gene_lengths, bins='auto')
    # plt.show()

    (k2, pval) = stats.mstats.normaltest(log_gene_lengths)
    print(k2)
    print(pval)

    filename = 'data/counts.txt'
    data_table =pd.read_csv(filename, index_col=0)
    print(data_table.iloc[:5, :5])
    counts = data_table.values
    log_counts = np.log(counts + 1)
    #plot_col_density(log_counts)

    log_counts_normalized = quantile_norm_log(log_counts)
    #plot_col_density(log_counts_normalized)

    counts_log = np.log(counts + 1)

    data_table = pd.read_csv(r"C:\Users\visu4\Documents\wcgna\data\counts.csv", index_col=0)
    print(data_table.iloc[:5, :5])
    print("Length of data table**********")
    print(len(data_table))
    print("*********Number of cols in data frame")
    print(len(data_table.columns))
    counts_var = most_variable_rows(counts_log, n=1500)
    yr, yc = bicluster(counts_var, linkage_method='ward', distance_metric='euclidean')

    #plot_bicluster(counts_var, yr, yc)

    n_clusters = 3
    threshold_distance = (yc[-n_clusters, 2] + yc[-n_clusters+1, 2])/2
    clusters =  fcluster(yc, threshold_distance, 'distance')

    # patients survival
    patients = pd.read_csv(r"C:\Users\visu4\Documents\wcgna\data\patients.csv.txt", index_col=0)
    patients.head()

    plot_cluster_survival_curves(clusters, data_table.columns, patients)
    ##########################################################

    # binned_boxplot(x=log_gene_lengths, y=mean_log_counts)

    # RPKM
    # C = counts
    # N = counts.sum(axis=0)
    # L = gene_lengths
    # L = L[:, np.newaxis]
    # C_tmp = (10^9 * C) / L
    # rpkm_counts = C_tmp / N[np.newaxis, :]
