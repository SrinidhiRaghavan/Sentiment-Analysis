# Sentiment-Analysis

The process of computationally identifying and categorizing opinions expressed as texts, is known as Sentiment Analysis. It is usually used to find how users feel about a particular topic. Say you want to find if the Indians in New York find Junoon good or bad, then Sentiment Analysis of the various reviews of Junoon can computationally answer this question. The applications of Sentiment Analysis / Opinion mining/ Text-mining is immense in the domain of computing customer satisfaction metrics. 

This project is a simpler version of Sentiment Analysis of the reviews of various restaurants in Pittsburgh. Note that the data (reviews_tr.csv) used here can be obtained online. This data comprises of the reviews of restaurants in Pittsburgh, along with a customer assigned rating on a scale of 5. You can use anyother data instead of reviews. However, note that the data must be in CSV format where the first line is the header, with the first column as "labels" and the second as "text". Additionally, preprocess the data to remove any non-alphanumeric symbols and make sure that all the characters are lowercase. We use three different representations for understanding the training data: Unigram, TF-IDF, Bigram and N-Gram. The definitions of all these representations is as given:
* Unigram Representation:
  In this representation, a feature is associated with every word in the document. The feature value associated with a word 'w' in the document 'd' defined as the number of times the word 'w' appears in the document 'd'. This feature value is also known as term frequency (tf) and is denoted by tf(w,d) 
  
* TF-IDF Representation:
  TFIDF stands for term-frequency inverse-document frequency weighting. This representation is a numeric statistic intended to reflect how important a word is in a document. The TFIDF of a word 'w' in document 'd' is given by: tf(w,d) * log(idf(w,D)). Here, log is computed with respect to base 10 and D is the training data which can be thought as a collection of documents. IDF(w,D) is defined as the ratio of the total number of documents in the training set and the number of documents in the training set that contains the word w. 
  
  Consider the training data to be the reviews given by two customers. Then, the number of documents in the training set is 2, and each document in the training set is the review given by each of these customers. 

* Bigram Representation:
  Every pair of words (w1,w2) in the document is known as bigram. In this representation, a feature is associated with all the bigrams in the document. The feature value is defined as the number of times the bigram occurs in the document.
  
* N-Gram Representation:
  The n-gram representation is defined as the number of times a set of n-words occurs together in the document 
  
In this project, we use Averaged perceptron and Naive Bayes classifier are used as the learning methods. Additionally, five-fold cross-validation is used to enhance the performance. Using these representations and learning methods, it was concluded that Averaged Perceptron Classification on Bigram Representation  was the best for the given text analysis 
