import itertools
from itertools import chain, combinations
from scipy.spatial import KDTree
import numpy as np
import pandas as pd


def dbscan(data, k, eps, minpts):
    c = 0



def main():
    data = pd.read_csv('iris.data', header=None)
    k = 4
    #  print(df.columns)
    # Added empty column for labels
    data[len(data.columns)] = np.nan
    dataA = pd.DataFrame(pd.np.random.rand(100, 3))
    dataA[3] = np.nan
    print(dataA.head())
    print(dataA.values)
    print("*******")
    print(dataA.iloc[0,0:3].values)



    # tree = KDTree(dataA.values)
    # ind = tree.query_ball_point(dataA[:1], r=0.3) #returns a list if only on epoint is queried
    
    tree = KDTree(dataA.iloc[:,0:3].values)
    ind = tree.query_ball_point(dataA.iloc[1,0:3], r=0.3) #returns a list if only on epoint is queried


    print("+++++++++++++++")
    print(ind)
    print(type(ind))
    print(len(ind))
    print("+++++++++++++++")
    print("+++++++++++++++")
    print(dataA[:1])

    # for i in ind[0]:
    #     print(dataA.iloc[i])
    #     print("--")
    #     print(np.linalg.norm(dataA[:1]-dataA.iloc[i]))
    

    for i in ind:
        print(dataA.iloc[i,0:3])
        print("--")
        print(np.linalg.norm(dataA.iloc[1,0:3]-dataA.iloc[i,0:3]))
    
if __name__ == "__main__":
    main()


