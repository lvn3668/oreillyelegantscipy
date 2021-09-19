# Author lalitha viswanathan
#  laplacian with brain data (worm brain data neuron network)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
# from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy import linalg


def plot_connectome(x_coords: object, y_coords: object, conn_matrix: np, *, labels=(), types=None, type_names=('',), xlabel='', ylabel=''):
    """

    :param x_coords:
    :param y_coords:
    :param conn_matrix:
    :param labels:
    :param types:
    :param type_names:
    :param xlabel:
    :param ylabel:
    """
    if types is None:
        types = np.zeros(x_coords.shape, dtype=int)
    ntypes: int = len(np.unique(types))
    colors = plt.rcParams['axes.prop_cycle'][:ntypes].by_key()['color']
    cmap: ListedColormap = ListedColormap(colors)
    fig, ax = plt.subplots()
    for neuron_type in range(ntypes):
        plotting = (types == neuron_type)
        pts = ax.scatter(x_coords[plotting], y_coords[plotting], c=cmap(neuron_type), s=4, zorder=1)
        pts.set_label(type_names[neuron_type])

    for x, y, label in zip(x_coords, y_coords, labels):
        ax.text(x, y, ' ' + label, verticalalignment='center', fontsize=3, zorder=2)
    pre, post = np.nonzero(conn_matrix)
    links = np.array([x_coords[pre], x_coords[post], y_coords[pre], y_coords[post]]).T
    print(links.T.shape)
    print(conn_matrix)
    #plt.show()
    # ax.add_collection(LineCollection(links.T, color='lightgray', lw=0.3, alpha=0.5, zorder=0))
    ax.legend(scatterpoints=3, fontsize=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    plt.show()

# download worm neuron data
connectome_url = 'http://www.wormatlas.org/images/NeuronConnect.xls'
conn = pd.read_excel(connectome_url)

conn_edges = [(n1, n2, {'weight': s}) for n1, n2, t, s in conn.itertuples(index=False, name=None) if t.startswith('S')]
wormbrain = nx.DiGraph()
wormbrain.add_edges_from(conn_edges)
centrality = nx.betweenness_centrality(wormbrain)
central = sorted(centrality, key=centrality.get, reverse=True)
print(central[:5])
sccs = nx.strongly_connected_components_recursive(wormbrain)
giantscc = max(sccs, key=len)
# print(f'The largest strongly connected component has '
#      f'{giantscc.number_of_nodes()} nodes out of '
#      f'{wormbrain.number_of_nodes()} total')

# in_degrees = list(wormbrain.in_degree().values())
# in_deg_distrib = np.bincount(in_degrees)
# avg_in_degree = np.mean(in_degrees)
# cumulativefreq= np.cumsum(in_deg_distrib) / np.sum(in_deg_distrib)
# survival = 1 - cumulativefreq

# fig, ax = plt.subplots()
# ax.loglog(np.arange(1, len(survival)+1), survival)
# ax.set_xlabel(' in degree distribution')
# ax.set_ylabel('fraction of neuron with higher in-degree distribution')
# ax.scatter(avg_in_degree,0.0022, marker='v')
# ax.text(avg_in_degree-0.5, 0.003, 'mean=%.2f' %avg_in_degree)
# ax.set_ylim(0.002, 1.0)


Chem = np.load(r"C:\Users\visu4\Documents\wcgna\data\chem-network.npy", allow_pickle="True", encoding="latin1")
Gap = np.load(r"C:\Users\visu4\Documents\wcgna\data\gap-network.npy")


# for labeling the n/w
neuron_ids = np.load(r"C:\Users\visu4\Documents\wcgna\data\neurons.npy", allow_pickle="True", encoding="latin1")
neuron_types = np.load(r"C:\Users\visu4\Documents\wcgna\data\neuron-types.npy", allow_pickle="True", encoding="latin1")
A = Chem + Gap # (neurons and link between neurons)
C = (A + A.T) / 2
# calculate degrees of freedom
degrees = np.sum(((Chem + Gap) + (Chem + Gap).T) / 2, axis=0)
D = np.diag(degrees)
# laplace xform to find most significant neurons
L = D - C
b = np.sum(C * np.sign(A - A.T), axis=1)
z = linalg.pinv(L) @ b # Matrix multiplication to find positive or negative correlation
Dinv2 = np.diag(1 / np.sqrt(degrees))
Q = Dinv2 @ L @ Dinv2 # inverse L again
val, Vec = linalg.eig(Q) # Eigen on Laplace (gives 2nd smallest eigen value : fiedler vector) -> tightly interconnected neurons
smallest_first = np.argsort(val)

#### smallest sub network ###
val = val[smallest_first]
Vec = Vec[:, smallest_first]
x = Dinv2 @ Vec[:, 1]
###

# sanity check to find VCB2 neuron and its connections
vc2_index = np.argwhere(neuron_ids == 'VCB2')
if vc2_index.size > 0:
    x = -x

plot_connectome(x, z, ((Chem + Gap) + (Chem + Gap).T) / 2, labels=neuron_ids, types=neuron_types,
                type_names=['sensory neurons', 'interneurons', 'motor neurons'], xlabel='Affinity eigenvector 1',
                ylabel='Processng depth')
