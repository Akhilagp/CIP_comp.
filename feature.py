import re
import os
import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from collections import Counter
from functools import reduce
import array

#import dataset

class FeatureExtractor(object):
	"""docstring for ClassName"""
	def __init__(self, domain_list):
		self._domain_list = domain_list
		self._positive_domain_list = None
		self._hmm_model_path = './models/hmm.pkl'
		self._big_grame_model_path = './models/big_grame.pkl'
		self._triple_grame_model_path = './models/tripple_gram.pkl'
		self._positive_grame_model_path = './models/positive_grame.pkl'
		self._word_grame_model_path = './models/word_grame.pkl'
		self._positive_count_matrix = './models/positive_count_matrix.npy'
		self._word_count_matrix = './models/word_count_matrix.npy' 
		self._positive_domain_list = self._load_positive_domain()
		#print(self._domain_list[:10])
	# check wether required files exis 
	def _check_files(self, *args):
		for val in args:
			if not os.path.exists(val):
				raise ValueError("file{} doesn't exis, check scripts \
					dataset and prepare_model ".format(val))


	def _load_positive_domain(self):
		positive = pd.read_csv('../datas/aleax100k.csv', names=['domain'], header=None, dtype={'word': np.str}, encoding='utf-8')
		positive = positive.dropna()
		positive = positive.drop_duplicates()
		return positive['domain'].tolist()

	def count_aeiou(self):
		count_result = []
		for domain in self._domain_list:
			len_aeiou = len(re.findall(r'[aeiou]',domain.lower()))
			aeiou_rate = (0.0+len_aeiou)/len(domain)
			tmp = [domain, len(domain), len_aeiou, aeiou_rate]
			count_result.append(tmp)

		count_result = pd.DataFrame(count_result, 
									columns=['domain','domain_len',
											 'aeiou_len','aeiou_rate'])
		return count_result


	#the rate between original domain length and seted domain length
	
	def unique_char_rate(self):
		unique_rate_list = []
		for domain in self._domain_list:
			unique_len = len(set(domain))
			unique_rate = (unique_len+0.0)/len(domain)
			tmp = [domain, unique_len, unique_rate]
			unique_rate_list.append(tmp)

		unique_rate_df = pd.DataFrame(unique_rate_list, 
										columns=['domain','unique_len','unique_rate'])
		return unique_rate_df


	# calculate double domain's jarccard index
	def _jarccard2domain(self, domain_aplha, domain_beta):
		"""parameters:
		domain_alpha/beta: string-like domain
		returns: this couples jarccard index
		"""
		listit_domain_alpha = list(domain_aplha)
		listit_domain_beta = list(domain_beta)

		abs_intersection = np.intersect1d(listit_domain_alpha, listit_domain_beta).shape[0]
		abs_union = np.union1d(listit_domain_alpha, listit_domain_beta).shape[0]
		
		return abs_intersection/abs_union*1.0


	# calculate each fake domain's average corresponding jarccard index 
	# with positive domain collection
	def jarccard_index(self):
		"""parameters:
		positive_domain_list: positve samples list, 1Darray like
		return: a pandas DataFrame, 
				contains domian col and average jarccard index col
		"""
		
		positive_domain_list = np.random.choice(self._positive_domain_list,500)
		#print("pdt: ",positive_domain_list)
		positive_domain_list = positive_domain_list.tolist()
		#print("pdt: ",positive_domain_list)

		jarccard_index_list = []
		for fake_domain in self._domain_list:
			total_jarccard_index = 0.0
			for std_domain in positive_domain_list:
				total_jarccard_index += self._jarccard2domain(fake_domain, std_domain)
			
			avg_jarccard_index = total_jarccard_index/len(positive_domain_list)
			tmp = [fake_domain, avg_jarccard_index]
			jarccard_index_list.append(tmp)

		jarccard_index_df = pd.DataFrame(jarccard_index_list, 
											columns=['domain','avg_jarccard_index'])

		return jarccard_index_df
	
	'''
	def levenshtein_distance(self,domain_alpha,domain_beta,mx=-1):
		def result(d): 
			return d if mx < 0 else False if d > mx else True
		if domain_alpha == domain_beta:
			return result(0)
		la, lb = len(domain_alpha), len(domain_beta)
		if mx >= 0 and abs(la - lb) > mx:
			return result(mx+1)
		if la == 0:
			return result(lb)
		if lb == 0:
			return result(la)
		if lb > la:
			domain_alpha, domain_beta, la, lb = domain_beta, domain_alpha, lb, la
 
		cost = array.array('i', range(lb + 1))
		for i in range(1, la + 1):
			cost[0] = i
			ls = i-1
			mn = ls
			for j in range(1, lb + 1):
				ls, act = cost[j], ls + int(domain_alpha[i-1] != domain_beta[j-1])
				cost[j] = min(ls+1, cost[j-1]+1, act)
				if (ls < mn):
					mn = ls
			if mx >= 0 and mn > mx:
				return result(mx+1)
		if mx >= 0 and cost[lb] > mx:
			return result(mx+1)
		return result(cost[lb])

	def levenshtein_distance_param(self):
		levenshtein_distance_list=[]
		for domain_index in range(1,len(self._domain_list)):
			levenshtein_val = self.levenshtein_distance(self._domain_list[domain_index-1],self._domain_list[domain_index])
			tmp = [self._domain_list[domain_index],levenshtein_val]
			levenshtein_distance_list.append(tmp)
		levenshtein_distance_df = pd.DataFrame(levenshtein_distance_list,columns=['domain','levenshtein_distance'])
		
		return levenshtein_distance_df
	'''

	def findValidWords(self):
		meaningful_word_ratio_list = []
		english_vocab = set(line.strip() for line in open('wordlist.txt','r'))
		for st in self._domain_list:
			tot=0.0
			all_words = {st[i:j + i] for j in range(2, len(st)) for i in range(len(st)- j + 1)}
			int_words = all_words.intersection(english_vocab)
			for i in int_words:
				if not any([i in sub_str for sub_str in int_words if i != sub_str]):
					tot+=len(i)
			meaningful_word_ratio = tot/len(st)
			tmp = [st, meaningful_word_ratio]
			meaningful_word_ratio_list.append(tmp)
		
		meaningful_word_ratio_df = pd.DataFrame(meaningful_word_ratio_list, columns=['domain','meaningful_word_ratio'])
		return meaningful_word_ratio_df
	


	def _domain2vec(domain):
		ver = []
		for i in range(0, len(domain)):
			ver.append([ord(domain[i])])
		return ver
	def positive_train(big_domain):
        	"""parameters:
        	big domain: large scale posotive domains, type:list
        	"""
        	vec = CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
        	grame_model = vec.fit(big_domain)
        	joblib.dump(grame_model, './models/positive_grame.pkl')
        	counts_matrix = grame_model.transform(big_domain)
        	positive_counts = np.log10(counts_matrix.sum(axis=0).getA1())
        	np.save('./models/positive_count_matrix.npy' , positive_counts)

	def word_train(word):
        	vec = CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
        	grame_model = vec.fit(word)
        	joblib.dump(grame_model, './models/word_grame.pkl')
        	counts_matrix = grame_model.transform(word)
        	word_counts =  np.log10(counts_matrix.sum(axis=0).getA1())
        	np.save('./models/word_count_matrix.npy', word_counts)


	# TODO
	def hmm_index(self):
		hmm_model = joblib.load(self._hmm_model_path)
		hmm_index_list = []
		for domain in self._domain_list:
			vec = self._domain2vec(domian)
			hmm_score = hmm_model.predict(vec)
			tmp = [domain, hmm_score]
			hmm_index_list.append(tmp)

		hmm_index_list = pd.DataFrame(hmm_index_list, columns=['domain','hmm_index'])

		return hmm_index_list


	'''
	# calculate n_grame of each domains
	# notes: you should update this model frequency
	# 		decrease the dimension of the transformed data set
	# 		and rank features
	def big_grame(self):
		if not os.path.exists(self._n_grame_model_path):
			raise("n_grame model dosen't exists, try to training this model\n\
				train scripts at same level folder called prepare_model.py\n\
				notes: training n_grame model by domains as much as you have")

		grame_model = joblib.load(self._n_grame_model_path)
		vec = grame_model.transform(np.array(self._domain_list))

		df = pd.DataFrame(vec.toarray(), columns=grame_model.get_feature_names())
		domains = pd.DataFrame(self._domain_list, columns=['domain'])
		df = pd.concat([domains, df], axis=1)
		
		return df

	def tripple_gram(self):
		pass
	
	'''


	#calculate entropy of domains entropy
	def entropy(self):
		"""parameters

		return: entropy DataFrame [doamin, entropy]
		"""
		entropy_list = []
		for domain in self._domain_list:
			p, lns = Counter(domain), float(len(domain))
			entropy = (-sum(count/lns * math.log(count/lns, 2) for count in p.values()))
			tmp = [domain, entropy]
			entropy_list.append(tmp)

		entropy_df = pd.DataFrame(entropy_list, columns=['domain','entropy'])
		return entropy_df



	#calculate grame(3,4,5) and its differ
	def n_grame(self):
		"""
		return local grame differ with positive domains and word list
		"""
		self._check_files(self._positive_count_matrix,
						  self._positive_grame_model_path,
						  self._word_grame_model_path,
						  self._word_count_matrix)
		
		positive_count_matrix = np.load(self._positive_count_matrix)
		positive_vectorizer = joblib.load(self._positive_grame_model_path)
		word_count_matrix = np.load(self._word_count_matrix)
		word_vectorizer = joblib.load(self._word_grame_model_path)

		positive_grames = positive_count_matrix * positive_vectorizer.transform(self._domain_list).T
		word_grames = word_count_matrix * word_vectorizer.transform(self._domain_list).T
		diff = positive_grames - word_grames
		domains = np.asarray(self._domain_list)


		n_grame_nd = np.c_[domains, positive_grames, word_grames, diff]
		n_grame_df = pd.DataFrame(n_grame_nd, columns=['domain','positive_grames','word_grames','diff'])
		
		return n_grame_df

	def digit_ratio(self):
		digit_pattern = r'\d'
		character_pattern = r'[A-Za-z]'
		character_finder = re.compile(character_pattern)
		digit_finder = re.compile(digit_pattern)
		digit_ration_list = []

		for domain in self._domain_list:
			digit_len = len(digit_finder.findall(domain))
			character_len = len(character_finder.findall(domain))
			domain_len = len(domain)
			digit_ratio = (digit_len+0.0)/domain_len
			character_ratio = (character_len+0.0)/domain_len
			tmp = [domain, digit_ratio, character_ratio]
			digit_ration_list.append(tmp)
		
		digit_ration_df = pd.DataFrame(digit_ration_list, columns=['domain',
																   'digit_ration',
																   'character_ratio']
																   )

		return digit_ration_df




def get_feature(domain_list):
	extractor = FeatureExtractor(domain_list)

	print("extracting count_aeiou....")
	aeiou_df = extractor.count_aeiou()
	print("extracted count_aeiou, shape is %d\n" % aeiou_df.shape[0])

	print("extracting unique_rate....")
	unique_rate_df = extractor.unique_char_rate()
	print("extracted unique_rate, shape is %d\n" % unique_rate_df.shape[0])

	print("extracting jarccard_index....")
	jarccard_index_df = extractor.jarccard_index()
	print("extracted jarccard_index.....\n")

	print("extracting entropy....")
	entropy_df = extractor.entropy()
	print("extracted entropy, shape is %d\n"%entropy_df.shape[0])
	
	print("extracting n_grame....")
	n_grame_df = extractor.n_grame()
	print("extracted n_grame, shape is %d\n"%n_grame_df.shape[0])
	#levenshtein_distance_df = extractor.levenshtein_distance_param()
	meaningful_word_ratio_df= extractor.findValidWords()
	print("merge all features on domains...")
	multiple_df = [aeiou_df, unique_rate_df, 
				  entropy_df, jarccard_index_df,
				  n_grame_df,meaningful_word_ratio_df]

	df_final = reduce(lambda left,right: pd.merge(left,right,on='domain',how='left'), multiple_df)
	#print("df_final",df_final.dtypes)
	print("merged all features, shape is %d\n"%df_final.shape[0])
	# check df
	std_rows = aeiou_df.shape[0]
	df_final_rows = df_final.shape[0]

	if std_rows != df_final_rows:
		raise("row dosen't match after merged multiple_df")

	df_final = df_final.drop(['domain'],axis=1)
	df_final = df_final.round(3)
	
	return np.array(df_final)
