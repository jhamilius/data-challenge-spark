import loadFiles as lf

import exctract_terms as et
import preProcess as pp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


data,Y=lf.loadLabeled("./train")
#analyzer and preprocessor dont work together...one or the other
pp.proc(data) # we apply the preprocessing by ourselves
m = TfidfVectorizer(analyzer=et.terms)

tt = m.fit_transform(data)

rows,cols=tt.shape
print "number of features :" +str(len(m.get_feature_names())) #this is the same as the number of columns
#the function get_feature_names() returns a list of all the terms found 

print "non compressed matrix expected size:" + str(rows*cols*8/(1024*1024*1024))+"GB"

