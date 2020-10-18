import numpy as np
import pandas as pd
import assessment as assessment
import dbscan as dbscan
import matplotlib.pyplot as plt


def main():
    parameters = [(0.1, 6), (0.15, 6), (0.2, 6), (0.3, 6), (0.485, 6), (0.7, 6), (1, 6)]
    index = 0
    for parameter in parameters:
        plt.close('all')
        data = pd.read_csv('iris.data', header=None)
        result = dbscan.dbscan(data, 4, parameter[0], parameter[1])
        result[len(result.columns)] = result.index
        clusterDetatails(result, 5, 6, parameter[0], parameter[1])
        puri = assessment.purity(result, 5, 6)
        print(puri)
        sill = assessment.silhouettecofficient(result, 6, 4)
        print(sill[0])
        result = result.sort_values(5)
        result = result.reset_index(drop=True)
        # print(result.head())
        plt.figure()
        # result.plot.scatter(x=np.array(result.shape), y=len(result.columns) - 1)
        plotname = 'irisData.dbscan.sill.result.' + str(index) + '.png'
        plt.scatter(np.arange(result.shape[0]), result[len(result.columns) - 1])
        plt.savefig(plotname, dpi=300)
        index = index + 1


#
#
#
#
# Synthetic_300S_63N.csv
# Synthetic_400S_63N.csv
# Synthetic_500S_0N.csv
# Synthetic_500S_34N.csv
# Synthetic_500S_66N.csv
# Synthetic_500S_82N.csv
# Synthetic_500S_99N.csv
# Synthetic_600S_99N.csv
# Synthetic_700S_179N.csv
def clusterDetatails(data, gtlabel, clabel, eps, minpts):
    print("radius: " + str(eps) + " min points: " + str(minpts))
    labels = data[clabel - 1].unique()
    glabels = data[gtlabel - 1].unique()
    count = 0
    noiseClusters = 0
    for l in glabels:
        if l == 0:
            noiseClusters = noiseClusters + 1
        else:
            count = count + 1
    print("Ground Truth Clusters: " + str(count) + " Noise Clusters: " + str(noiseClusters))

    count = 0
    noiseClusters = 0
    for l in labels:
        if l == -1:
            noiseClusters = noiseClusters + 1
        else:
            count = count + 1
    print("Clusters: " + str(count) + " Noise Clusters: " + str(noiseClusters))


if __name__ == '__main__':
    main()
