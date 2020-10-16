import itertools
from itertools import chain, combinations

import numpy as np
import pandas as pd

from numpy.random import rand


def k_means(df, K, epsilon):

    D = df.copy()
    t = 0
    dim = len(D.columns)-1
    N_t = len(D.index)
    # print("Number of txn")
    # print (N_t)
    print("Number of classes")
    print(K)
    mu = np.zeros(shape=(K, dim))


    # print("Min Max range of input data")
    D_min = D.min().values
    D_max = D.max().values
    # print(D_min)

    # Initailization according to K-means++
    # ref : https://www.geeksforgeeks.org/ml-k-means-algorithm/
    for k in range(K):
        if k ==0 :
            mu[k, :] = D_min[0:dim]+(D_max[0:dim] - D_min[0:dim])*rand(dim)
        else:
            min_dist = np.ones(shape=(1, N_t))*100
            for j in range(N_t):
                xj = D.iloc[j].values
                for i in range(k):
                    err_k = xj[0:dim] - mu[i, :]
                    dist = np.linalg.norm(err_k, ord=2)**2
                    min_dist[0, j] = min(min_dist[0, j], dist)
            j_max = np.argmax(min_dist)
            xj_max = D.iloc[j_max].values
            mu[k, :] = xj_max[0:dim]


    print("Initial Mean")
    print(mu)


    while True:
        print("iteration")
        t = t+1
        print(t)

        C = pd.DataFrame(np.zeros(shape=(N_t, 1)), columns=["Kmeans_Class"])

        # print("Mu old")
        mu_old = mu.copy()
        # print(mu_old)

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

                err_k = xj[0:dim] - mu[k, :]
                # print("error in k")
                # print(err_k)
                res[0, k] = np.linalg.norm(err_k, ord=2)**2
                # print("residual")
                # print(res)

            i_star = np.argmin(res)+1
            # print("i_star")
            # print(i_star)

            C.iloc[j] = i_star

        # Centroid update
        for k in range(K):
            # print("Kth cluster k=")
            # print(k)
            idx_k = C.loc[C['Kmeans_Class']==k+1].index
            # print("Class indexes")
            # print(idx_k)

            if len(idx_k) != 0:
                c_k = D.iloc[idx_k.values]
                # print("C_k ")
                # print(c_k)

                # print("cluster mean")
                # print(c_k.mean())

                mu[k, :] = c_k.mean()[0:dim]

        # print("New mean")
        # print(mu)

        # print("Mean error")
        # print(np.subtract(mu, mu_old))
        #
        # print("Mean error norm")
        # print(np.linalg.norm(np.subtract(mu, mu_old), ord=2, axis=1)**2)
        print("SSE(mu)")
        print(sum(np.linalg.norm(np.subtract(mu, mu_old), ord=2, axis=1) ** 2))
        if sum(np.linalg.norm(np.subtract(mu, mu_old), ord=2, axis=1) ** 2) <= epsilon:
            print("Number of iterations")
            print(t)
            D[dim+1] = C
            return D
            break

def main():
    print('**************Hello*********')

    k = 3
    epsilon = 0.001
    epochs = 5

    data = pd.read_csv('iris.data', header=None)
    result = k_means(data, k, epsilon)
    result.to_csv('iris.data.result.k_means.csv', index=False, header=False)

    data2 = pd.read_csv('Synthetic_Data_Label.csv', header=None)
    result2 = k_means(data2, k, epsilon)
    result2.to_csv('sysnthetic.data.result.k_means.csv', index=False, header=False)





def test():
    df = pd.read_csv('iris.data', header=None)
    print(df.describe())
    # print(df[0])

    k = 3
    epsilon = 0.001

    print("Calling K means")
    k_means(df, k, epsilon)

    print("k means done")


if __name__ == "__main__":
    main()
