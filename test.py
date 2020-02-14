from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import LabelEncoder

def checkFileExist(filename):
    if os.path.isfile(filename):
        return True
    else:
        return False

def DownloadDataSet(_file_id, _des_path):
    gdd.download_file_from_goole_drive(file_id=_file_id, des_path=_des_path)

if checkFileExist('train.csv'):
    print("file has already existed, do not download")
else:
    gdd.download_file_from_google_drive(file_id='1SiWs-DQgE9cvu3iBq9QhddEfi8LBG2z6', dest_path='./train.csv')

if checkFileExist('test.csv'):
    print("file has already existed, do not download")
else:
    gdd.download_file_from_google_drive(file_id='1OgdQZ1UwGz-jvRzXW2bDPB3k8I501-TT', dest_path='./test.csv')

if checkFileExist('sample_sbumission.csv'):
    print("file has already existed, do not download")
else:
    gdd.download_file_from_google_drive(file_id='1LaH_bWe-OQE2pPA6XO3OFT8hB6RWrEdS',dest_path='./sample_submission.csv')

if checkFileExist('columns_description.csv'):
    print("file has already existed, do not download")
else:
    gdd.download_file_from_google_drive(file_id='1rb8wFMniWPAsd123UOyTtmlxuHcHWoTk',dest_path='./columns_description.csv')

train_df = pd.read_csv('train.csv', index_col='id')
print(train_df.shape)
train_df.head(15)

print(train_df['label'])

cat_features = ['province','district', 'maCv']
num_features = [col for col in train_df.columns if col not in cat_features]
print(num_features)

# test
train_df = train_df.dropna()
encoder = LabelEncoder()
prov = train_df['province']
prov_encoded = encoder.fit_transform(prov)
print(prov_encoded)
print(prov_encoded.shape)
print(encoder.classes_)
prov_encoded = prov_encoded.reshape(-1,1)
prov_encoded = prov_encoded.toarray()
print(prov_encoded)
