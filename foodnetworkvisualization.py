# Author: Elegant SciPy (most significant food consumption / food network visualization
# )
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def pagerank_plot(in_degrees: list, pageranks: np, names: list[str], *, annotations: list[str], **figkwargs: list):
    """

    :param pageranks: np
    :param in_degrees:
    :param names:
    :type figkwargs: object (keyword args for graph)
    """
    fig, ax = plt.subplots(**figkwargs)
    ax.scatter(in_degrees, pageranks, c=[0.835, 0.369, 0], lw=0)
    for name, indeg, pr in zip(names, in_degrees, pageranks):
        if name in annotations:
            text = ax.text(indeg + 0.1, pr, name)
    ax.set_ylim(0, np.max(pageranks) * 1.1)
    ax.set_xlim(-1, np.max(in_degrees) * 1.1)
    ax.set_ylabel('PageRank')
    ax.set_xlabel('In-degree')
    plt.show()


stmarks = nx.read_gml(r"C:\Users\visu4\Documents\wcgna\data\stmarks.gml.txt")
species = np.array(stmarks.nodes())

# adjacency matrix (if species a eats food b, entry [a,b] in adjacency matrix has non-zero value 
Adj = nx.to_scipy_sparse_matrix(stmarks, dtype=np.float64)
n = len(species)
#print("number of species in node ", n)
np.seterr(divide='ignore')
# Reduce the species interaction (adjacency matrix) to a sparse matrix and flatten the dimensions 
degrees = np.ravel(Adj.sum(axis=1))
Deginv = sparse.diags(1 / degrees).tocsr()

# xpose of degree inv X adjacency matrix
Trans = (Deginv @ Adj).T
damping = 0.85
beta = 1 - damping

# convert diagonals to 1s (compressed space column format for storing the matrix)
I = sparse.eye(n, format='csc')
# Solve AX = B ; A = sparse matrix (number of species) 
# np.full (convert the resulting sparse matrix to specified dimensions : number of species and fill value (beta/ no of species))
pagerank = spsolve(I - damping * Trans, np.full(n, beta / n))
#print(pagerank)
interesting = ['detritus', 'phytoplankton', 'benthic algae', 'micro-epiphytes', 'microfauna', 'zooplankton',
               'predatory shrimps', 'meiofauna', 'gulls']
# find most significant species-food interactions and return as flattened 1d array 
in_degrees = np.ravel(Adj.sum(axis=0))

pagerank_plot(in_degrees, pagerank, species, annotations=interesting)
