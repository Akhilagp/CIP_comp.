import os
import numpy as np
import pandas as pd

def load_simple_data():
	files = os.listdir('../AlgorithmPowereddomains')
	#print files	
	domain_list = []
	for f in files:
		path = '../AlgorithmPowereddomains/'+f
		domains = pd.read_csv(path,names=['domain'])
		domains = domains['domain'].tolist()
		for item in domains:
			domain_list.append(item)
	#print domain_list
	return domain_list


def load_data():
	#print("HEY")
	if os.path.exists('../datas/train.npy'):
		train = np.load('../datas/train.npy')
		return train
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
	#print "HI"	
	train = pd.read_csv("blah.csv",header=None,names=['domain','label'])
	#train1 = np.array(train)
	print(train.head(5))
	
	train1 = np.array(train.values)
	np.random.shuffle(train1)
	np.save('../datas/train.npy', train1)

	return train

	
#print "HI"
load_simple_data()
load_data()
