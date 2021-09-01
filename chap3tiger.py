# Image Processing
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd
from skimage import segmentation, color

def threshold_graph(g, t):
    to_remove = [(u,v) for(u,v,d) in g.edges(data=True) if d['weight'] > t]
    g.remove_edges_from(to_remove)

def add_edge_filter(values, graph):
    center = values[len(values) // 2]
    for neighbor in values:
        if neighbor != center and not graph.has_edge(center, neighbor):
            graph.add_edge(center, neighbor)
    return 0.0

def build_rag(labels, image):
    g = nx.Graph()
    nrows, ncols = labels.shape
    for row in range(nrows):
        for col in range(ncols):
            current_label = labels[row, col]
            if not current_label in g:
                g.add_node(current_label)
                g.node[current_label]['total color'] = np.zeros(3, dtype=np.float)
                g.node[current_label]['pixel count'] = 0
            if row < nrows - 1 and labels[row+1, col] != current_label:
                g.add_edge(current_label, labels[row+1, col])
            if col < ncols - 1 and labels[row, col+1] != current_label:
                g.add_edge(current_label, labels[row+1, col])
            g.node[current_label]['total color'] += image[row, col]
            g.node[current_label]['pixel count'] += 1
    return g

def build_rag(labels, image):
    g = nx.Graph()
    footprint = nd.generate_binary_structure(labels.ndim, connectivity=1)
    _ = nd.generic_filter(labels, add_edge_filter, footprint=footprint, node='nearest', extra_arguments=(g,))
    for n in g:
        g.node[n]['total color']= np.zeros(3, np.double)
        g.node[n]['pixel count']=0
    for index in np.ndindex(labels.shape):
        n = labels[index]
        g.node[n]['total color']+= image[index]
        g.node[n]['pixel count'] += 1
    return g

if __name__ == '__main__':
    print("Inside main")
    url = (
        'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg')
    tiger = io.imread(url)
    seg = segmentation.slic(tiger, n_segments=30, compactness=40.0,
                            enforce_connectivity=True, sigma=3, start_label=1)
    io.imshow(color.label2rgb(seg, tiger))
    io.show()
    g = build_rag(seg, tiger)
    for n in g:
        node = g.node[n]
        node['mean'] = node['total color'] / node['pixel count']
    for u,v in g.edges_iter():
        d = g.node[u]['mean'] - g.node[v]['mean']
        g[u][v]['weight'] = np.linalg.norm(d)
    threshold_graph(g, 80)
    map_array = np.zeros(np.max(seg)+1, int)
    for i, segment in enumerate(nx.connected_components(g)):
        for initial in segment:
            map_array[int(initial)] = i
    segmented = map_array[seg]
    plt.imshow(color.label2rgb(segmented, tiger))
    plt.show()

#def build_rag(labels, image):
#    g = nx.Graph()
#    nrows, ncols = labels.shape
#    for row in range(nrows):
#        for col in range(ncols):
#            current_label = labels[row, col]
#            if not current_label in g:
#                g.add_node(current_label)
#                g.node[current_label]['total color'] = np.zeros(3, dtype=np.float)
#                g.node[current_label]['pixel count'] = 0
#            if row < nrows - 1 and labels[row+1, col] != current_label:
#                g.add_edge(current_label, labels[row+1, col])
#            if col < ncols - 1 and labels[row, col+1] != current_label:
#                g.add_edge(current_label, labels[row+1, col])
#            g.node[current_label]['total color'] += image[row, col]
#            g.node[current_label]['pixel count'] += 1
#    return g

