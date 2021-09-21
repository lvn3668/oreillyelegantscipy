import numpy as np
from scipy import sparse
from scipy import ndimage as ndi
from skimage import color, segmentation
import matplotlib.pyplot as plt
from skimage import data, io
from itertools import product
import networkx as nx
import numpy as np
from skimage.future import graph


def add_edge_filter(values, graph):
    current = values[0]
    neighbors = values[1:]
    for neighbor in neighbors:
        graph.add_edge(current, neighbor)
    return 0

def build_rag(labels, image):
    g = nx.Graph()
    footprint = ndi.generate_binary_structure(labels.ndim, connectivity=1)
    for j in range(labels.ndim):
        fp = np.swapaxes(footprint, j, 0)
        fp[0, ...] = 0
    _ = ndi.generic_filter(labels, add_edge_filter, footprint=footprint, mode='nearest', extra_arguments=(g,))

    for n in g:
        g.node[n]['total color'] = np.zeros(3, np.double)
        g.node[n]['pixel count'] = 0

    for index in np.ndindex(labels.shape):
        n = labels[index]
        g.node[n]['total color'] += image[index]
        g.node[n]['pixel count'] += 1
    return g

def threshold_graph(g, t):
    to_remove = ((u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > t)
    g.remove_edges_from(to_remove)


def invert_nonzero(arr):
    arr_inv =  arr.copy()
    nz = np.nonzero(arr)
    arr_inv[nz]= 1/arr[nz]
    return arr_inv



def xlog1x(arr_or_mat):
    out = arr_or_mat.copy()
    if isinstance(out, sparse.spmatrix):
        arr = out.data
    else:
        arr = out
    nz = np.nonzero(arr)
    arr[nz] *= - np.log2(arr[nz])
    return out



def homography(tf, image_shape):
    H = np.linalg.inv(tf)
    m, n = image_shape
    row, col, values = [], [], []

    for sparse_op_row, (out_row, out_col) in enumerate(product(range(m), range(n))):
        in_row, in_col, in_abs = H @ [out_row, out_col, 1]
        in_row /= in_abs
        in_col /= in_abs

        if not 0 <= in_row < m-1 or not 0 <= in_col < n-1:
            continue

        top = int(np.floor(in_row))
        left = int(np.floor(in_col))
        t = in_row - top
        u = in_col - left
        row.extend([sparse_op_row]*4)
        sparse_op_col = np.ravel_multi_index(
            (
                [top, top, top+1, top+1],
                [left, left+1, left, left+1]
            ), dims =(m, n)
        )
        col.extend(sparse_op_col)
        values.extend([(1-t)*(1-u), (1-t)*u, t*(1-u), t*u])
    operator = sparse.coo_matrix((values, (row, col)), shape=(m*n, m*n)).tocsr()
    return operator

def apply_transform(image, tf):
    return (tf @ image.flat).reshape(image.shape)



def confusion_matrix(pred, gt):
    cont = np.zeros((2, 2))
    for i in [0, 1]:
        for j in [0, 1]:
            cont[i, j] = np.sum((pred == i) & (gt == j))
    return cont



def variation_of_information(x, y):
    n = x.size
    Pxy = sparse.coo_matrix((np.full(n, 1/n), (x.ravel(), y.ravel())), dtype=float).tocsr()
    px = np.ravel(Pxy.sum(axis=1))
    py = np.ravel(Pxy.sum(axis=0))
    Px_inv = sparse.diags(invert_nonzero(px))
    Py_inv = sparse.diags(invert_nonzero(py))
    hygx = px @ xlog1x(Px_inv @ Pxy).sum(axis=1)
    hxgy = xlog1x(Pxy @ Py_inv).sum(axis=0)@ py
    #return 0.0
    return float(hygx + hxgy)

def rag_segmentation(base_seg, image, threshold=80):
    g=build_rag(base_seg, image)
    for n in g:
        node = g.node[n]
        node['mean'] = node['total color'] / node['pixel count']
    for u, v in g.edges_iter():
        d = g.node[u]['mean'] - g.node[v]['mean']
        g[u][v]['weight'] = np.linalg.norm(d)

    threshold_graph(g, threshold)
    map_array =  np.zeros(np.max(seg)+1, int)
    for i, segment in enumerate(nx.connected_components(g)):
        for initial in segment:
            map_array[int(initial)] = i
    segmented = map_array[seg]
    return(segmented)

angle = 30
c = np.cos(np.deg2rad(angle))
s = np.sin(np.deg2rad(angle))
H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
point =  np.array([1, 0,1])
print(np.sqrt(3)/2)
print(H @ point)
print(H @ H @ H @ point)

print(" before displaying cameraman image")
image = data.camera()
#print(image)
plt.imshow(image)
plt.show()

tf = homography(H, image.shape)
out = apply_transform(image, tf)
plt.imshow(out)
plt.show()

url  = ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg')
tiger = io.imread(url)
plt.imshow(tiger)
plt.show()

human_seg_url = ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/human/normal/outline/color/1122/108073.jpg')
boundaries = io.imread(human_seg_url)
plt.imshow(boundaries)
plt.show()

seg = segmentation.slic(tiger, n_segments=30, compactness =40.0, enforce_connectivity=True, sigma=3)
plt.imshow(color.label2rgb(seg, tiger))
plt.show()

auto_seg_10 = rag_segmentation(seg, tiger, threshold=10)
plt.imshow(color.label2rgb(auto_seg_10, tiger))
plt.show()

auto_seg_40=rag_segmentation(seg, tiger, threshold=40)
plt.imshow(color.label2rgb(auto_seg_40,tiger))
plt.show()