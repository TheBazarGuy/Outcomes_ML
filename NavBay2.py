import numpy as np
import csv
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
from dats import X_train
from dats import y_train_text2





s1="engineering"
s2="problem analysis"
s3="design"
s4="research"
s5="moderntool"
s6="engineersociety"
s7="ennvironment"
s8="ethics"
s9="teamwork"
s10="communication"
s11="projectmanagement"
s12="learning"


ls1=[]
x_arr=[]
yy_train_text=[]
with open('codata.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		#print(' '.join(row))
		str11='$'.join(row)
		#print(str)
		ls1=str11.split("$")
		#print(ls1[0]+"  "+ls1[1])
		x_arr.append(ls1[0])
		str2=ls1[1].replace("[","")
		str2=str2.replace("]","")
		temp_list=str2.split(",")
		yy_train_text.append(temp_list)

XX_train=np.array(x_arr)



count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
print(X_train_counts.shape)
#print(count_vect.vocabulary_.get(u'algorithm'))
#print(X_train_counts)


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

y_sparse = sparse.csr_matrix(y_train_text2)
clf = MultinomialNB().fit(X_train_tfidf, y_train_text2)


#docs_new = ['Knuth-Morris-Pratt','engineering']
strl=input("enter string\n")
docs_new=[]
docs_new.append(strl)

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
	print(doc+"  "+"Po"+str(category))



























