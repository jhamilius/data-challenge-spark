import loadFiles as lf
import preProcess as pp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from  pyspark.mllib.regression import LabeledPoint
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import SparseVector
from pyspark import SparkContext
from pyspark import SparkFiles
from functools import partial


#a function that transforms the coordinate data and applies predict
#this is an alternative approach so that we may do one call to the executors
def predict(row_coord,cSize,model):
	vector_dict={}
	for w in row_coord[1]:
		vector_dict[int(w[1])]=w[0]
	return (row_coord[0], model.value.predict(SparseVector(cSize.value,vector_dict)))

trainF="./data/train" #the path to where the train data is
testF="./data/test" # the path to the unlabelled data 
saveF="./predictions.txt" #where to save the predictions

sc = SparkContext(appName="Simple App")  #initialize the spark context
#since we are not in the command line interface we need to add to the spark context
#some of our classes so that they are available to the workers
sc.addFile("/home/christos.giatsidis/data_camp_2015_dec/helpers.py") 
sc.addFile("/home/christos.giatsidis/data_camp_2015_dec/exctract_terms.py")
#now if we import these files they will also be available to the workers
from helpers import *
import exctract_terms as et



# load data : data is a list with the text per doc in each cell. Y is the respective class value
#1 :positive , 0 negative
print "loading local data"
data,Y=lf.loadLabeled(trainF) 

print "preprocessing"
pp.proc(data) #clean up the data from  number, html tags, punctuations (except for "?!." ...."?!" are replaced by "."
m = TfidfVectorizer(analyzer=et.terms) # m is a compressed matrix with the tfidf matrix the terms are extracted with our own custom function 

'''
we need an array to distribute to the workers ...
the array should be the same size as the number of workers
we need one element per worker only
'''
ex=np.zeros(8) 
rp=randint(0,7)
ex[rp]=1 #one random worker will be selected so we set one random element to non-zero

md=sc.broadcast(m) #broadcast the vectorizer so that he will be available to all workers
datad=sc.broadcast(data) # broadcast the data

#execute vectorizer in one random  remote machine 
#partial is a python function that calls a function and can assign partially some of the parameters 
#numSlices determins how mnay partitions should the data have
#numslices is also helpfull if we want to reduce the size of each task for each worker
tmpRDD=sc.parallelize(ex,numSlices=8).filter(lambda x: x!=0).map(partial(compute, model=md, data=datad))
print "transforming the data in a remote machine"
data=tmpRDD.collect() # get back the coordinate matrix and the fitted vectorizer
#data = [[[matrix][vectorizer]]] (double nested)
tfidf_coo=data[0][0] 
m=data[0][1] #the fitted vectorizer re-assign it just in case

datad.unpersist() # we don't need this broadcasted variable anymore




#distribute the coordinate data 
# data =[ [value,row_index,column_index],[value,row_index,column_index]..]
ttodp=sc.parallelize(tfidf_coo,512) 

# a function to combine 2 lists into one list
def comb(x,y): 
	x.extend(list(y))
	return x
	
#organize the coordinate matrix into the row index and a tuple containing the value and column index
#group by the row index 
tmp=ttodp.map(lambda x: (x[1],[(x[0],x[2])])).aggregateByKey([],comb,comb)

bY=sc.broadcast(Y) #broadcast the class variable (in order to create labeled points)
# the number of features is the columns of the matrix
#we need this information to convert to vectors and label point the coordinate data
cols=sc.broadcast(len(m.get_feature_names())) 
print "number of features"+str(cols.value)

#convert to labeled point in parallel
tmpLB=tmp.map(partial(createLabeledPoint,cSize=cols,classes=bY)) 
print "training the machine learning algorithm"
model_trained = NaiveBayes.train(tmpLB) # train a naive bayes model
mtBR=sc.broadcast(model_trained)

print "loading unlabeled data"
test,names=lf.loadUknown(testF) #load the unlabelled data . test : text per document. names : the respective file names
namesb=sc.broadcast(names) # broadcast the file names as we will need them for predictions

md=sc.broadcast(m) # broadcast the fitted model of the vecotrizer 
datadt=sc.broadcast(test) # broadcast the unlabeled data so that we may call the vectorizer in same manner



#apply the vectorization in a random worker
print "transforming unlabelled data"
test_data=sc.parallelize(ex,numSlices=16).filter(lambda x: x!=0).map(partial(computeTest, model=md,data=datadt)).collect()


datadt.unpersist()
print "convert data to non-compressed vector and predict the class"


#Steps:
#distribute the coordinate tf-idf of the transformed unlabelled data
#organize the coordinate by row
##aggregate the data by row
#convert the coordinate data to a vector &
#apply for each vector the prediction and return the prediction along with the name of the file for that prediction
#(predict does both of the two last steps)
test_data_d=sc.parallelize(test_data[0],numSlices=512).map(lambda x: (x[1],[(x[0],x[2])])).aggregateByKey([],comb,comb).map(partial(predict,cSize=cols,model=mtBR))

predictions=test_data_d.collect() # get all the (filename, prediction) tuples 
print "writing prediction"
#write the predictions to a file
f=open(saveF,'w')
for x in predictions:
    f.write(names[int(x[0])])
    f.write(',')
    f.write(str(x[1]))
    f.write('\n')
f.close()