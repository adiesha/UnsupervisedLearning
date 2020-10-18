import numpy as np
import pandas as pd
import assessment as assessment
import dbscan as dbscan


def main():
    parameters = [(0.3, 6), (0.485, 6), (0.7, 6), (1, 6)]
    for parameter in parameters:
        data = pd.read_csv('iris.data', header=None)
        result = dbscan.dbscan(data, 4, parameter[0], parameter[1])
        result[len(data.columns)] = data.index
        puri = assessment.purity(result, 5, 6)
        print(puri)
        sill = assessment.silhouettecofficient(result, 6, 4)
        print(sill)


if __name__ == '__main__':
    main()
