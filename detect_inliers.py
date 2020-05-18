## This code corresponds section A.4 in the paper.
import itertools
import numpy as np


## TO-DO  Figure-out if this involves 3d points or 2d points ?

def get_pairwise_distance(features):
    pair = np.asarray(list(itertools.product(l,l))).reshape(len(l),len(l),2,2)
    x1 = pair[:, :, 0, 0]
    y1 = pair[:, :, 0, 1]
    x2 = pair[:, :, 1, 0]
    y2 = pair[:, :, 1, 1]
    ## Use the following code for debugging.
    print(np.column_stack((x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten())))
    dist = ((x1-x2)**2 + (y1-y2)**2)
    return dist




if __name__=="__main__":
    l = [[1, 2], [3, 4], [5, 6]]

    print("Pair-wise distances are \n",get_pairwise_distance(l))