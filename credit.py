import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('train.csv', index_col='id', low_memory=False)
print(train_df.shape)

# 1. imputation
# numerical imputation
# categorical imputation
