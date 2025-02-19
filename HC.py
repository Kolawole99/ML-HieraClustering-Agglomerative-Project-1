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
import matplotlib.cm as acm
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

#==============================agglomerative function from scikit learn to cluster===========================
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_

#=======================add a new field to our dataframe to show the cluster they belong to==================
pdf['cluster_'] = agglom.labels_
pdf.head()

#=============================================plotting the cluster image=================================
n_clusters = max(agglom.labels_)+1
colors = acm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))
for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

#========================Summarizing and distributing clusters using the vehicle type===============
pdf.groupby(['cluster_','type'])['cluster_'].count()
#--viewing the clusters---
agg_cars = pdf.groupby(['cluster_','type'])["horsepow","engine_s","mpg","price"].mean()
agg_cars


#=======================We plot again using the new distribution set-up===================
plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()
