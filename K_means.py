import itertools
from itertools import chain, combinations

import numpy as np
import pandas as pd

from numpy.random import rand


def k_means(df,K,epsilon):

    D = df.copy()
    t = 0
    dim = len(D.columns)
    N_t = len(D.index)
    print("Number of txn")
    print (N_t)
    mu = np.zeros(shape=(K,dim))

    for k in range(K):

        mu[k,:]  = rand(4)
    #      a + (b-a) * random().


    print("Mean")
    print(mu)

    print("Min Max range of input data")
    # D_minmax = pd.Series(index=['min','max'],data=[D.min(),D.max()])
    D_minmax = [D.min().values,D.max().values]
    print(D_minmax)




    # mu_new = np.array()
    while True:
        t = t+1

        C = pd.DataFrame(np.zeros(shape=(N_t, 1)), columns=["Clac Class"])

        # Cluster Assignment step
        for j in range(N_t) :
            # print("Calculating for j =")
            # print(j)

            xj = D.iloc[j].values

            # print("X_j values")
            # print(xj[0:dim])


            res = np.zeros(shape=(1,K))
            for k in range(K):
                # print("Calculating for class k=")
                # print(k)
                # print("mean value")
                # print(mu[k])

                err_k = xj[0:dim] - mu[k,:]
                # print("error in k")
                # print(err_k)
                res[0,k] = np.linalg.norm(err_k,ord=2)**2
                # print("residual")
                # print(res)

            i_star = np.argmin(res)+1
            # print("i_star")
            # print(i_star)

            C[j] = i_star


        for k in range(K):
            mu[k,:] = mu[k,:]

        if np.linalg.norm([mu_new - mu], ord='fro',axis=0)<= epsilon:
            break

def main():
    print('**************Hello*********')


    df = pd.read_csv('iris.data', header=None)
    print(df.describe())
    # print(df[0])

    k = 3
    epsilon = 0.001

    print("Calling K means")
    k_means(df,k,epsilon)

    print("k means done")
if __name__ == "__main__":
    main()
