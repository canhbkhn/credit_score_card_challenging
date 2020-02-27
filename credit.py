import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

class DataProc:
    # load data
    def load_data(self,data_path):
        return pd.read_csv(data_path, low_memory=False)

    # imputation data
    def imputation(self, df, imputationType, columns):
        if imputationType == "numerical":
            print("--> numerical imputation")
            return df[columns].fillna(df[columns].median())
        else:
            print("--> categorical imputation")

    # handling outliers
    def handling_outliers(self):
        pass

    # binning
    def binning_data(self):
        pass

    # log transform
    def log_transform(self):
        pass

    # one-hot encoding
    def one_hot_encoding(self):
        pass

    # grouping operations
    def grouping_operation(self):
        pass

    # feature split
    def feature_split(self):
        pass

    # scaling
    def scaling(self):
        pass

    # extracting date
    def extracting_date(self):
        pass

class training:
    # traning
    def training_data(self):
        pass
    # predict
    def prediction(self):
        pass

#train_df = pd.read_csv('train.csv', index_col='id', low_memory=False)
# print(train_df['FIELD_55'])
#print(train_df.shape)
#print(train_df['maCv'].head(50))
#print(train_df['FIELD_52'].head(50))

# 1. imputation
# numerical imputation
# numerical = ['FIELD_52', 'FIELD_55']
#train_df['FIELD_55'] = train_df['FIELD_55'].fillna(train_df['FIELD_55'].median())
#print(train_df['FIELD_55'])

#train_df['FIELD_52'] = train_df['FIELD_52'].fillna(0, inplace=True)
# categorical imputation

#print(train_df['FIELD_52'])
# categorical_imputaion = []
# 2. handling outlier
# 3. binning
# numerical binning
# categorical binning ['maCv']
#conditions = [
#        train_df['maCv'].str.contains('Công nhân'),
#        train_df['maCv'].str.contains('Nhân viên'),
#        train_df['maCv'].str.contains('Cấp dưỡng')
#        ]
#choices = ['Công nhân', 'Nhân viên', 'Cấp dưỡng']
#choices = ['CN', 'NV', 'CD']
#train_df['maCv'] = np.select(conditions, choices, default='Other')
#print(train_df['maCv'])
# 4. log transform(logarithm transformation)
# 5. one-hot encoding
# 6. grouping operations
# 7. feature split
# 8. scaling
# 9. extracting date
#

if __name__ == '__main__':
    train_df = DataProc()
    data = train_df.load_data('train.csv')
    data['FIELD_55'] = train_df.imputation(data, "numerical", 'FIELD_55')
    print(data)
   # clf = LogisticRegression()
