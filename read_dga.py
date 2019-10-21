import pandas as pd
l = []
with open("train_data.txt",'r') as f:
	for line in f:
		i = line.strip("\n").split(" ")
		l.append(i)

df = pd.DataFrame(l)
df.to_csv("blah.csv",header=None,index=False)
