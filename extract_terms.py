import re
import nltk.data 
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pyspark.ml.feature import NGram
from nltk.collocations import *

def terms(doc):
	Terms=[]
	sentences=sent_tokenize(doc.decode("utf-8")) # break the document into sentences could be useful for further extensions
	for sentence in sentences :
		Terms.extend(sentence_terms(sentence.lower())) # add the terms found per sentence

	#Adding Bigrams and Trigrams
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	# trigram_measures = nltk.collocations.TrigramAssocMeasures()

	finder2 = BigramCollocationFinder.from_words(Terms)
	finder2.apply_freq_filter(3)
	bigrams = finder2.nbest(bigram_measures.pmi, 20)

	# finder3 = TrigramCollocationFinder.from_words(Terms)
	# finder3.apply_freq_filter(3)
	# trigrams = finder3.nbest(trigram_measures.chi_sq, 200)

	Terms.extend(bigrams)
	# Terms.extend(trigrams)

	return Terms
#returns the terms found in one sentence
def sentence_terms(sentence) :
	#sentence=str(sentence.encode("utf-8"))
	stop_words=stopwords.words('english') # get a list of default stopwords
	sentence=re.sub('[?!\\.]+','',sentence).strip() #remove punctuation from sentence because we don't need it anymore
	sentence=re.sub('\s+',' ',sentence)	# remove multiple spaces
	stemmer = PorterStemmer()
	# split the sentence into words
	#if the word is not in the list of stopwords apply stemming on it
	#and then add it to the list. If aterm appears twice then it will be added twice
	terms=[stemmer.stem(w) for w in sentence.split(" ") if w not in stop_words] 
	return terms


