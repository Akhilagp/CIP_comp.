import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

ar = np.load(sys.argv[2],allow_pickle=True)
#print(ar)
ar_1 = ar[ar[:,-1]==1]
#print(ar_1)
ar_1 = ar_1[:,:-2]
#print(ar_1)
inp_type =ar_1[:,-2:]
#print(inp_type,type(inp_type))

'''
for i in [x*0.5 for x in range(1,10)]:
	for j in range(3,15):
		print("i,j",i,j)
		dbscan = DBSCAN(eps=i, metric='euclidean', min_samples=j).fit(ar_1)
		#pca = PCA(n_components=2).fit(ar_1)
		#print("pca",pca)
		#pca_2d = pca.transform(ar_1)
		print(np.unique(dbscan.labels_))
'''
#dbscan = DBSCAN(eps=4.5, metric='euclidean', min_samples=10).fit(ar_1)
dbscan = OPTICS(min_samples=15,cluster_method='xi').fit(ar_1)
pca = PCA(n_components=3).fit(ar_1)
pca_2d = pca.transform(ar_1)
print(np.unique(dbscan.labels_))
y_pred = dbscan.fit_predict(pca_2d)
op = np.concatenate((pca_2d,y_pred[:,None]),axis=1)
tt = np.concatenate((y_pred[:,None],inp_type),axis=1)
np.save(sys.argv[1],tt)
print(op.shape)
op = op[op[:,-1]!=-1]
y_pred = op[:,-1]
plt.scatter(op[:,0], op[:,1],300,c=y_pred, cmap='Paired',label=y_pred)
plt.show()

'''
for i in range(0, pca_2d.shape[0]):
	if dbscan.labels_[i] == 0:
		c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
	elif dbscan.labels_[i] == 1:
		c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
	elif dbscan.labels_[i] == -1:
		c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
	elif dbscan.labels_[i] == 2:
		c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='o')

pl.legend([c1, c2, c3, c4], ['Cluster 1', 'Cluster 2','Noise','cc'])
pl.title('DBSCAN finds 2 clusters and noise')
pl.show()

4., 6
4.5 9


# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

X = ar_1[:,:3]

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=4, min_samples=6, metric='euclidean').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
labels_true = inp_type[:,1]
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f     % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
#print("Silhouette Coefficient: %0.3f"      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    #xy = X[class_member_mask & core_samples_mask]
    #print(xy)
    for i in range(0,X.shape[0]):
        plt.scatter(X[i,1], X[i, 2], marker='*',c = tuple(col))

    #xy = X[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
     #        markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
'''
