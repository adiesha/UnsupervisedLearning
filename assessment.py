import pandas as pd


def main():
    data = pd.read_csv('iris.data.result.csv', header=None)
    data[len(data.columns)] = data.index
    assessment(data, 5, 6)


def test():
    print("Test")
    data = pd.read_csv('iris.data.result.csv', header=None)
    unique = data[4].unique()
    print(unique)
    data[len(data.columns)] = data.index
    data.to_csv('test.scv', index=False, header=False)
    assessment(data, 5, 6)


def assessment(data, truthColumn, labelColumn):
    externamIndex = purity(data, truthColumn, labelColumn)


def purity(data, truthColumn, labelColumn):
    # create the ground truth lists

    indexColumn = len(data.columns) - 1
    groundTruthList = data.groupby(truthColumn - 1)[indexColumn].apply(list).values.tolist()
    print(groundTruthList)
    groundTruthListCount = len(groundTruthList)
    print(groundTruthListCount)

    # create the cluster list
    clusterList = data.groupby(labelColumn - 1)[indexColumn].apply(list).values.tolist()
    print(clusterList)
    clusterListCount = len(groundTruthList)
    print(clusterListCount)

    puritySum = 0
    for i in clusterList:
        iSet = set(i)
        count = 0
        for j in groundTruthList:
            jSet = set(j)
            temp = iSet & jSet
            if temp and len(temp) > count:
                count = len(temp)
        puritySum = puritySum + count

    purity = puritySum / data.shape[0]
    print(purity)
    return purity


if __name__ == "__main__":
    main()
