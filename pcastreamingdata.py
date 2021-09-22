import matplotlib.pyplot as plt
import toolz as tz
from toolz import curried as c
from sklearn import decomposition
from sklearn import datasets
import numpy as np

def streaming_pca(samples, n_components=2, batch_size=100):
    ipca = decomposition.IncrementalPCA(n_components=n_components, batch_size=batch_size)
    tz.pipe(samples, c.partition(batch_size), c.map(np.array), c.map(ipca.partial_fit), tz.last)
    return ipca

reshape = tz.curry(np.reshape)

def array_from_txt(line, sep=',', dtype=np.float):
    return np.array(line.rstrip().split(sep), dtype=dtype)

with open(r"C:\Users\visu4\Documents\wcgna\data\iris.csv.txt") as fin:
    pca_obj = tz.pipe(fin, c.map(array_from_txt), streaming_pca)

with open(r"C:\Users\visu4\Documents\wcgna\data\iris.csv.txt") as fin:
    components = tz.pipe(fin, c.map(array_from_txt), c.map(reshape(newshape=(1, -1))), c.map(pca_obj.transform), np.vstack)
    print(components.shape)

iris_types = np.loadtxt(r"C:\Users\visu4\Documents\wcgna\data\iris-target.csv.txt")
plt.scatter(*components.T, c=iris_types, cmap='viridis')

iris = np.loadtxt(r"C:\Users\visu4\Documents\wcgna\data\iris-target.csv.txt", delimiter=',')
components2 = decomposition.PCA(n_components=2).fit_transform(iris)
plt.scatter(*components2.T, c=iris_types, cmap='viridis')
