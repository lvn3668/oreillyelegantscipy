import base64

import networkx as nx
import np as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy import ndimage as ndi
from skimage import morphology
import pandas as pd
from skimage import segmentation
from skimage import color

def tax(prices):
    return 10000+0.05*np.percentile(prices, 90)


def add_edge_filter(values, graph):
    center = values[ len(values) // 2]
    for neighbor in values:
        if neighbor != center and not graph.has_edge(center, neighbor):
            graph.add_edge(center, neighbor)
    return 0.0

def reduce_xaxis_labels (ax, factor):

    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
    for label in ax.xaxis.get_ticklabels()[::factor]:
        label.set_visible(True)

def build_rag(labels, image):
    g = nx.Graph()
    footprint = ndi.generate_binary_structure(labels.ndim, connectivity=1)
    _= ndi.generic_filter(labels, add_edge_filter, footprint=footprint, node='nearest', extra_arguments=(g,))
    return g

def overlay_grid(image, spacing=128):
    image_gridded = image.copy()
    pass
    return image_gridded

def gaussian_kernel (size, sigma):
    positions = np.arange(size) - size // 2
    kernel_raw = np.exp(-positions**2 / (2 * sigma**2))
    kernel_normalized = kernel_raw / np.sum(kernel_raw)
    return kernel_normalized


if __name__ == '__main__':
    print("Image processing Chap 3")
    random_image = np.random.rand(500,500)
    url_coins = ('https://raw.githubusercontent.com/scikit-image/scikit-image/v0.10.1/skimage/data/coins.png')
    coins = io.imread(url_coins)
    url_astronaut = ('https://raw.githubusercontent.com/scikit-image/scikit-image/master/skimage/data/astronaut.png')
    astro = io.imread(url_astronaut)
    print("Type:", type(coins), "Shape:", coins.shape, "Data-type:", coins.dtype)

    plt.imshow(random_image)
    plt.imshow(coins)
    astro_nq = np.copy(astro)
    astro_nq[50:100, 50:100] = [0,255,0] # RGB  channel (r=0, g=255, b=0)
    sq_mask = np.zeros(astro.shape[:2], bool)
    sq_mask[50:100, 50:100] = True
    astro_nq[sq_mask] = [0,255,0]
    sig = np.zeros(100, np.float)
    sig[30:60] = 1
    fig, ax = plt.subplots()
    ax.plot(sig)
    ax.set_ylim(-0.1,1.1)
    sigdelta = sig[1:]
    sigdiff = sigdelta - sig[:-1]
    sigon = np.clip(sigdiff, 0, np.inf)
    fig, ax = plt.subplots()
    ax.plot(sigon)
    ax.set_ylim(-0.1, 1.1)
    print('Signal on at ', 1+np.flatnonzero(sigon)[0], 'ms')
    plt.imshow(astro_nq)
    diff = np.array([1,0,-1])
    np.random.seed(0)
    sig = sig + np.random.normal(0,0.3,size=sig.shape)
    plt.plot(sig)
    plt.plot(ndi.convolve(sig, diff))

    smooth_diff = ndi.convolve(gaussian_kernel(25,3), diff)
    plt.plot(smooth_diff)
    sdsig = ndi.convolve(sig, smooth_diff)
    plt.plot(sdsig)
    plt.imshow(overlay_grid(astro, 128))
    plt.show()

    coins = coins.astype(float) / 255
    diff2d = np.array([[0,1,0],[1,0,-1], [0,-1,0]])
    coins_edges = ndi.convolve(coins, diff2d)
    io.imshow(coins_edges)
    io.show()

    hsobel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    vsobel = hsobel.T
    coins_h = ndi.convolve(coins, hsobel)
    coins_v = ndi.convolve(coins, vsobel)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(coins_h, cmap=plt.cm.RdBu)
    axes[1].imshow(coins_v, cmap=plt.cm.RdBu)
    for ax in axes:
        reduce_xaxis_labels(ax, 2)
    coins_sobel = np.sqrt(coins_h**2, coins_v**2)
    plt.imshow(coins_sobel, cmap='viridis')
    plt.show()
    house_price_map = (0.5+np.random.rand(100,100))*0.000001
    footprint = morphology.disk(radius=10)
    tax_rate_map = ndi.generic_filter(house_price_map, tax, footprint=footprint)
    plt.imshow(tax_rate_map)
    plt.colorbar()
    plt.imshow
    plt.show()

    connectome_url = 'http://www.wormatlas.org/images/NeuronConnect.xls'
    conn = pd.read_excel(connectome_url)
    # Conn edges contains only chemical synapses ; Every connectome other than type S is filtered out
    # weight is a keyword
    conn_edges = [(n1, n2, {'weight': s}) for n1, n2, t, s in conn.itertuples(index=False, name=None) if t.startswith('S')]
    wormbrain = nx.DiGraph()
    wormbrain.add_edges_from(conn_edges)

    centrality = nx.betweenness_centrality(wormbrain)
    central = sorted(centrality, key=centrality.get, reverse=True)
    print(central[:5])

    # Unable to find strongly connected components
    sccs = nx.algorithms.strongly_connected_components_recursive(wormbrain)
    giantscc = max(sccs, key=len)
    in_degrees = list((wormbrain.in_degree([0,1])).degree().values())
    in_deg_distrib = np.bincount(in_degrees)
    avg_in_degree = np.mean(in_degrees)
    cumfreq= np.cumsum(in_deg_distrib) / np.sum(in_deg_distrib)
    survival = 1- cumfreq

    fig, ax = plt.subplots()
    ax.loglog(np.arrange(1, len(survival)+1), survival)
    ax.set_xlabel('in degree distribution')
    ax.set_ylabel('fraction of neurons with higher in degree dist')
    ax.scatter(avg_in_degree, 0.0022, marker='v')
    ax.text(avg_in_degree -0.5, 0.003, 'mean=%.2f' %avg_in_degree)
    ax.set_ylim(0.002,1.0)

    url = ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg')
    tiger = io.imread(url)
    seg = segmentation.slic(tiger, n_segments=30, compactness=40.0, enforce_connectivity=True, sigma=3)
    io.imshow(color.lab2rgb(seg, tiger))
    io.show()