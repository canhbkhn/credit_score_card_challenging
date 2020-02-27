import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# grouping session

data_path = 'train.csv'
train_df = pd.read_csv(data_path,index_col='id', low_memory=False)

print(train_df.shape)

print(train_df['province'])

print(train_df['province'].value_counts())

cat_feature_count = train_df['province'].value_counts()
value_counts_list = cat_feature_count.index.tolist()
# print(value_counts_list)
# su dung binning cho truong thong tin nay
# 0-500-1000-2500 low-mid-high
#train_df['bin'] = pd.cut(train_df['province'], bins=[0,500,1000,2500], labels=["Low","Mid", "High"])
train_df['province'] = train_df['province'].replace(train_df['province'], train_df['province'].value_counts())

# imputation numerical: age_source1, age_source2
train_df['age_source1'] = train_df['age_source1'].fillna(train_df['age_source1'].median())
#print(train_df['age_source1'].describe())

train_df['age_source2'] = train_df['age_source2'].fillna(train_df['age_source2'].median())
#print(train_df['age_source2'])

# numericals feature
numericals_feature = ['FIELD_3', 'FIELD_4', 'FIELD_5', 'FIELD_6',
'FIELD_11', 'FIELD_16', 'FIELD_22', 'FIELD_45', 'FIELD_50', 'FIELD_51',
'FIELD_52', 'FIELD_53', 'FIELD_55', 'FIELD_56', 'FIELD_57']

print(train_df['FIELD_3'].describe())
train_df['FIELD_3'] = (train_df['FIELD_3'] - train_df['FIELD_3'].min()) / (train_df['FIELD_3'].max() - train_df['FIELD_3'].min())

print(train_df['FIELD_3'])
#for num in numericals_feature:
#	for ii in range(0, 30000):
#		if train_df[num][ii] == 'None' or train_df[num][ii] == '':
#			train_df[num][ii] = 0

# one-hot encoding field 4 5 6
print('--> processing field_4')
print(train_df['FIELD_4'].describe())
for ii in range(0,30000):
	if train_df['FIELD_4'][ii] == '':
		train_df['FIELD_4'][ii] == 0
	if train_df['FIELD_4'][ii] > 1:
		train_df['FIELD_4'][ii] = 1
print(train_df['FIELD_4'].describe())
print(train_df['FIELD_4'])

print('--> processing field_5')
print(train_df['FIELD_5'].describe())
for ii in range(0,30000):
	if train_df['FIELD_5'][ii] == '':
		train_df['FIELD_5'][ii] == 0
	if train_df['FIELD_5'][ii] > 1:
		train_df['FIELD_5'][ii] = 1
print(train_df['FIELD_5'].describe())
print(train_df['FIELD_5'])

print('--> processing field_6')
print(train_df['FIELD_6'].describe())
for ii in range(0,30000):
	if train_df['FIELD_6'][ii] == '':
		train_df['FIELD_6'][ii] == 0
	if train_df['FIELD_6'][ii] > 1:
		train_df['FIELD_6'][ii] = 1
print(train_df['FIELD_6'].describe())
print(train_df['FIELD_6'])

# field_8
print('--> processing field_8')
print(train_df['FIELD_8'].describe())
#print(train_df['FIELD_8'].count())
for ii in range(0, 30000):
	if train_df['FIELD_8'][ii] == '' or train_df['FIELD_8'][ii] == "FEMALE":
		train_df['FIELD_8'][ii] = 0
	else:
		train_df['FIELD_8'][ii] = 1
print(train_df['FIELD_8'])

# field_16: fill none and normalization
print('--> processing field_16')
print(train_df['FIELD_16'].describe())
for ii in range(0,30000):
	if train_df['FIELD_16'][ii] == '' or train_df['FIELD_16'][ii] != 1:
		train_df['FIELD_16'][ii] = 0
print(train_df['FIELD_16'])

# field_17
print('--> processing field_17')
print(train_df['FIELD_17'].describe())
for ii in range(0,30000):
	if train_df['FIELD_17'][ii] == "G8":
		train_df['FIELD_17'][ii] = 1
	else:
		train_df['FIELD_17'][ii] = 0
print(train_df['FIELD_17'])

# field_18,19,20
#cate_bool = ['FIELD_18', 'FIELD_19', 'FIELD_20']

print('--> processing field_18')
for ii in range(0,30000):
	if train_df['FIELD_18'][ii] == "TRUE":
		train_df['FIELD_18'][ii] = 1
	else:
		train_df['FIELD_18'][ii] = 0
print(train_df['FIELD_18'])

print(train_df['FIELD_19'].describe())