import loadFiles as lf
import preProcess as pp
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from  pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import Word2Vec
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.linalg import SparseVector
from pyspark import SparkContext
from pyspark import SparkFiles
from functools import partial
from pyspark.ml.feature import NGram


trainF="./data/train" #the path to where the train data is

sc = SparkContext(appName=" \--(o_o)--/ ")  #initialize the spark context


#since we are not in the command line interface we need to add to the spark context
#some of our classes so that they are available to the workers
sc.addFile("/home/julien.hamilius/datacamp/code/helpers.py") 
sc.addFile("/home/julien.hamilius/datacamp/code/extract_terms.py")
#now if we import these files they will also be available to the workers
from helpers import *
import extract_terms as et


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

rows=sc.broadcast(len(Y)) 
comb2 = (lambda x,y : np.vstack([np.array(x),np.array(y)]))
tmp2=ttodp.map(lambda x : (x[2],(x[0],x[1]))).aggregateByKey([0,0],comb2,comb2).map(partial(toVector, cols=rows)).map(lambda x : (x[0],chi2(np.array([x[1],x[1]]).T,np.array(bY.value)))).map(lambda x: (x[0],x[1][0][1])).collect()

print "features selection"
df = pd.DataFrame(tmp2) # transformer la liste de paires en dataframe   
df = df.sort_values(by=[1], ascending=False) # trier le dataframe par valeur de chi2 descendante
#PARAM
nbfeat = 100000 #len(col_index)
col_index = df.head(nbfeat)[0].values
print "- nb total de col : "+str(len(df))
print "- poids total des chi2 :"+str(df[1].sum(axis=0))
print "- nb de col selectionnees : "+str(nbfeat)
print "- poids selectionne :"+str(df[1][0:nbfeat].sum(axis=0))

tmp=ttodp.filter(lambda x: x[2] in col_index).map(lambda x: (x[1],[(x[0],x[2])])).aggregateByKey([],comb,comb)


print "number of features"+str(cols.value)

#convert to labeled point in parallel
tmpLB=tmp.map(partial(createLabeledPoint,cSize=cols,classes=bY)) 



print "splitting the data"
train, test = tmpLB.randomSplit([0.6, 0.4], seed = 0)
print "training the machine learning algorithm"
model = LogisticRegressionWithLBFGS.train(train, iterations=100, initialWeights=None,regParam=0.01, regType='l2', intercept=True, corrections=10, tolerance=0.0001, validateData=True, numClasses=2)



print "retrieving predictions and evaluating"
predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
print "accuracy:"+str(accuracy)