import os
import numpy as np



def load(path):

	rootdirPOS =path+'/pos'
	rootdirNEG =path+'/neg'
	names={}
	for subdir, dirs, files in os.walk(rootdirPOS):
		
		for file in files:
			with open(rootdirPOS+"/"+file, 'r') as content_file:
				names[file.split(".")[0]]=1
	
	for subdir, dirs, files in os.walk(rootdirNEG):
		
		for file in files:
			with open(rootdirNEG+"/"+file, 'r') as content_file:
				names[file.split(".")[0]]=0
	
	
	return names
	
def accu(real,predictions):
	correct=0
	for x in predictions:
		if(x[0] in real):
			if(real[x[0]]==float(x[1])): correct=correct+1
		else:
			print ":"+x[0]
	return float(correct)/float(len(real))
	
pathP="./predictions.txt"
pathR="./test_n"
real=load(pathR)
pred=[]
with open(pathP, 'r') as content_file:
	pred=[x.strip().split(",") for x in  content_file.read().split("\n")]
print str(accu(real,pred))