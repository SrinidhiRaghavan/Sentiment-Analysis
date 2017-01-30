import pandas as pd
import numpy as np
import time as time 

start=time.time()
training_data=pd.read_csv("reviews_tr.csv",header=0) #Reading the training CSV file
test_data=pd.read_csv("reviews_te.csv",header=0) #Reading the test CSV file

training_labels=training_data['label'] #Extracting the labels of the training dataset
training_reviews=training_data['text'] #Extracting the reviews of the training dataset

test_labels=test_data['label'] #Extracting the labels of the test dataset
test_reviews=test_data['text'] #Extracting the reviews of the test dataset


#Finding the unigram representation
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
unigram_training_words=vectorizer.fit_transform(training_reviews[0:200000])

print unigram_training_words.shape




#Finding the tf-idf representation
from sklearn.feature_extraction.text import TfidfTransformer 
transformer=TfidfTransformer(norm=None,smooth_idf=False,sublinear_tf=False,use_idf=True)
tfidf_training_words=transformer.fit_transform(unigram_training_words)-unigram_training_words
print (tfidf_training_words)




# #Finding the bigram representation 
bigram_vectorizer=CountVectorizer(ngram_range=(1,2))
bigram_training_words=bigram_vectorizer.fit_transform(training_reviews[0:1000])
print (bigram_training_words.shape)




#Additional Representations
#N-Gram
ngram_vectorizer=CountVectorizer(ngram_range=(1,3))
ngram_training_words=ngram_vectorizer.fit_transform(training_reviews)
print (ngram_training_words.shape)

#Modified IDF
transformer=TfidfTransformer(norm=None,smooth_idf=True,sublinear_tf=False,use_idf=True)
modified_tfidf_training_words=transformer.fit_transform(unigram_training_words)
print (modified_tfidf_training_words)




#Averaged Perceptron
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import safe_sparse_dot
def perceptron(x, y):
    y=np.where(y==0,-1,y)
    
    w=np.zeros(x.shape[1])   
    b=0
    
    n=x.shape[0]
    for i in range(n):
        a=safe_sparse_dot(x[i],w.T)+b

        if y[i]*a<=0:
            w=w+np.multiply(x[i],y[i])
            b=b+y[i]
    return w,b

def averaged_perceptron(x,y,w,b):
    u=w
    beta=b
    c=1
    
    y=np.where(y==0,-1,y)
    
    n=x.shape[0]
    for i in range(n):
        a=safe_sparse_dot(x[i],w.T)+b

        if y[i]*a<=0:
            w=w+np.multiply(x[i],y[i])
            b=b+y[i]
            u=u+c*np.multiply(x[i],y[i])
            beta=beta+c*y[i]
    
        c=c+1
    
    w=w-1/float(c)*u
    b=b-1/float(c)*beta
    return w,b

def perceptron_classifier(x_training,y_training,x_test,y_test):
    w,b=perceptron(x_training,y_training)
    w,b=averaged_perceptron(x_training,y_training,w,b)
    
    n=x_test.shape[0]
    y_predicted=np.zeros(x_test.shape[0])
    
    y_predicted=safe_sparse_dot(x_test,w.T)+b
    y_predicted=np.reshape(y_predicted,(n,1))
    y_test=np.reshape(y_test,(n,1))
    y_predicted[y_predicted<=0]=0
    y_predicted[y_predicted>0]=1
    
    error=np.sum(y_test!=y_predicted)
    error=error*100
    error=error/float(n)
    return error




#Cross-Validation
from sklearn.model_selection import KFold

kf=KFold(n_splits=5)

for training_index, test_index in kf.split(bigram_training_words):
   X_training, X_test = bigram_training_words[training_index], bigram_training_words[test_index]
   Y_training, Y_test = training_labels[training_index], training_labels[test_index]

   print perceptron_classifier(X_training, Y_training, X_test, Y_test)

print (time.time()-start)




#Naive-Bayes 
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

for training_index, test_index in kf.split(unigram_training_words):
    X_training, X_test = unigram_training_words[training_index], unigram_training_words[test_index]
    Y_training, Y_test = training_labels[training_index], training_labels[test_index]

    classifier= mnb.fit(X_training, Y_training)
    Y_predicted= classifier.predict(X_test)
    
    n=X_test.shape[0]
    error=np.sum(Y_test!=Y_predicted)
    error=error*100
    error=error/float(n)
    print error
    
print (time.time()-start)
