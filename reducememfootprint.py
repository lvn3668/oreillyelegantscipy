import numpy as np
from scipy import sparse
def confusion_matrix(pred, gt):
    n = pred.size
    ones = np.broadcast_to(1., n)
    cont = sparse.coo_matrix((np.ones(pred.size), (pred, gt)))
    return cont


pred = [0, 0,2]
gt = [1, 1, 2]
cont = confusion_matrix(pred, gt)
print(cont.toarray())
