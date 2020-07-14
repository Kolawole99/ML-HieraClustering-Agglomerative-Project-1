#=====================================IMPORTING NEEDED LIBRARIES=====================================
import numpy as np 
import pandas as pd
import scipy
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
#%matplotlib inline #useful in jupyter notebooks



#==========================================READING THE DATA=============================================
filename = 'cars_clus.csv'
#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)
df = pdf.head(5)
print(df)



#============================================DATA PROCESSING=============================================

#=============================================data cleaning=======================================
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

#===========================================feature selection===================================
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

#============================================normalization===================================
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]



#========================================CLUSTERING USING SCIKIT-LEARN=========================================
dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)

