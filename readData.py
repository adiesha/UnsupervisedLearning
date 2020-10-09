import itertools
from itertools import chain, combinations

import numpy as np
import pandas as pd

def main():
     df = pd.read_csv('iris.data', header=None)
     print(df.describe())
     print(df[0])

if __name__ == "__main__":
    main()

