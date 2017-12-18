import numpy as np
import csv
from dats import X_train
from dats import y_train_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer




s1="po1"
s2="po2"
s3="po3"
s4="po4"
s5="po5"
s6="po6"
s7="po7"
s8="po8"
s9="po9"
s10="p10"
s11="p11"
s12="p12"


ls1=[]
x_arr=[]
yy_train_text=[]
with open('codata.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		#print(' '.join(row))
		str='$'.join(row)
		#print(str)
		ls1=str.split("$")
		#print(ls1[0]+"  "+ls1[1])
		x_arr.append(ls1[0])
		str2=ls1[1].replace("[","")
		str2=str2.replace("]","")
		temp_list=str2.split(",")
		yy_train_text.append(temp_list)

XX_train=np.array(x_arr)






#y_train_text=[["identify"],["review"],["identify","review"]]

str=input("Enter string \n")
lst=[]
lst.append(str)

X_test=np.array(lst)
#X_test=np.array(['practical applications','I/O devices'])

#print(X_train)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print('{0} => {1}'.format(item, ', '.join(labels)))

