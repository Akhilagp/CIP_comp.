import os
import pandas as pd
import numpy as np
if os.path.exists('../datas/train.npy'):
	train = np.load('../datas/train.npy')

	'''
	domains360 = pd.read_csv('../datas/3601.txt',header=None)[[1]]
	domains360 = domains360.dropna()
	domains360['label'] = [0]*domains360.shape[0]
	print(domains360.head(10))


	#domains360 = domains360.drop_duplicates()


	domainsdga = pd.read_csv('../datas/dga-feed.txt', 
			names=['domain'], 
			header=None)
	domainsdga = domainsdga.dropna()
	domainsdga['label'] = [0]*domainsdga.shape[0]

	domain_normal = pd.read_csv('../datas/normal_domains.csv', 
			names=['domain'],
			header=None)
	domain_normal = domain_normal.dropna()
	domain_normal['label'] = [1]*domain_normal.shape[0]


	train = np.concatenate((domains360.values, domainsdga.values, domain_normal.values),axis=0)
	'''
	#train = train.drop_duplicates(subset=1)
print "HI"
train = pd.read_csv("blah.csv",header=None,names=['domain','label'])
#train1 = np.array(train)
print train.head(10)
#print train.values

