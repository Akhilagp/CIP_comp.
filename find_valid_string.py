import nltk
import sys
def findValidWords(domain):
	english_vocab = set(line.strip() for line in open('wordlist.txt','r'))
	for st in domain:
		#st = st[0]
		print(st[0])
		tot=0.0
		all_words = {st[0][i:j + i] for j in range(2, len(st[0])) for i in range(len(st[0])- j + 1)}
		int_words = all_words.intersection(english_vocab)
		print("int",int_words)
		for i in int_words:
			if not any([i in sub_str for sub_str in int_words if i != sub_str]):
				#print(i)
				tot+=len(i)
		print(st[0],st[1],tot/len(st[0]))

li=[]
with open('blah.csv','r') as f:
	for line in f:
		li.append(line.strip("\n").split(','))

findValidWords(li[0:50])		
#findValidWords(['gbtconnect.com','cutyxkktiycruemem.cc','myproject.com','linkedin.com','ensogakadhai.com','ingislamuniver.ru','kosmas.de','jqizzjcniqe.biz'])
#print(sys.argv[1])
