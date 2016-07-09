import numpy as np
from  pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector

'''
This file contains functions to help convert arrays and matrices in pySpark
friendly structures
It also contains function that execute remotely code so that the main server
will not get congested
'''
################################################
'''
The matrix we get as result from the tfidf is not really stored in rows and columns but in a compressed format 
Many machine learning models and evaluation functions will require vectorized data in order to classify/predict. 
This is useful for when we want to transform the compressed tf-idf matrix for the unlabeled data.
This function receives x where x is in the format [row,[[value,column],[value,column]...]
where :
row  is the row in matrix 
[value,column] : the value that corresponds to the "column"
'''
def toVector(x,cols) :
 t=np.zeros(cols.value)
 for c in x[1]:
  t[int(c[1])]=c[0]
 return (x[0],t)
############################################

'''
the spark Machine learning  library is optimized for Labeled points : (label/class, vector)
This function receives x where x is in the format (row_index,[(value,column_index),(value,column_index)...])
where :
row_index  represents the position of the data (assuming a matrix of labeled points; it is the row of the matrix) 
(value,column_index) : the value that corresponds to the column of that "column_index"

RETURNS: a labeledpoint : (label, sparse vector)
where sparse vector is the row in the matrix without the zeros values

#a function to create labeled point from coordinate information#
#row_coord: coordinate information per row. (row_index,[(value,column_index),(value,column_index)...]
#cSize: the number of columns
#classes the list containing the classes for all rows . we can use this to access class at row_index(row_coord[0])
'''
def createLabeledPoint(row_coord,cSize,classes):
	# set a dictionary {column_index:value} from the coord infomation (row_coord[1])
	#we can use that to build a SparseVector
	vector_dict={}
	for w in row_coord[1]:
		vector_dict[int(w[1])]=w[0]
	class_value=classes.value[row_coord[0]]

	return LabeledPoint(class_value, SparseVector(cSize.value,vector_dict))
############################################ 
'''
this function is used  compute the tfidf compressed matrix 
It assumes the following broadcaster variables :
md: the broadcasted TfidfVectorizer
datad: the broadcasted data
RETURNS: the matrix in coordinated format and the vectorizer that has been fitted on the data
coordinate format :
3 arrays [data,row,cols] transposed
the row and cols arrays contain the coordinates where the matrix has non-zero values
and the data array has the values e.g. data[i]=matrix[row[i],cols[i]]
THE transposed version of this matrix contains at each row [data_value,row_index,column_index]
'''
def compute(ar,model,data):
 tt = model.value.fit_transform(data.value)
 tt=tt.tocoo()
 tt=np.vstack([tt.data,tt.row,tt.col]) 
 tt=tt.transpose()
 return (tt,model.value)
'''
this function is the same as compute but we do not fit the vectorizer 
on the test data (it is already fitted)  and we do not return it (the vectorizer)
It assumes the following broadcaster variables :
md: the broadcasted FITTED TfidfVectorizer 
datad: the broadcasted  UNLABELED data
'''
def computeTest(ar,model,data):
 tt = model.value.transform(data.value)
 tt=tt.tocoo()
 tt=np.vstack([tt.data,tt.row,tt.col]) 
 tt=tt.transpose()
 return tt 
 

 

