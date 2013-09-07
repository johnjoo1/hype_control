from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import pylab as pl
import nltk
import sys
import pickle

from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from random import shuffle

class Model(object):
	def list_article_text(self, fname, label):
		f = open(fname, 'r')
		text_list = []
		text=[]
		target=[]
		for line in f:
			if line.startswith('headline:'):
				text = ' '.join(text)
				text_list.append(text)
				target.append(label)
				text=[]
			elif line.startswith('byline:'):
				pass
			elif line.startswith('8:31'):
				pass
			else:
				text.append(line)
		f.close()
		return text_list, target

	def merge_txt_lists(self, text_lists, same_proportion=True, randomize = True):
		merged = []
		min_num_texts = min([len(text) for text in text_lists])
		if same_proportion == True:
			for text_list in text_lists:
				merged += text_list[:min_num_texts]
		else:
			for text_list in text_lists:
				merged += text_list
		shuffle(merged)
		return merged

	def reload_raw_data(self, pickle_flag=False):
		text_list_news_dom, target_news_dom = self.list_article_text('./training_set/reuters_domestic_stories.txt', 0)
		text_list_news_dom = list(set(text_list_news_dom))
		target_news_dom = list(np.zeros(len(text_list_news_dom)))
		# text_list_news_dom, target_news_dom = strip_stopwords(text_list_news_dom, target_news_dom)

		text_list_news_world, target_news_world = self.list_article_text('./training_set/reuters_world_stories.txt', 0)
		text_list_news_world = list(set(text_list_news_world))
		target_news_world = list(np.zeros(len(text_list_news_world)))
		# text_list_news_world, target_news_world = strip_stopwords(text_list_news_world, target_news_world)

		text_list_news = self.merge_txt_lists([text_list_news_dom, text_list_news_world])
		target_news = list(np.zeros(len(text_list_news)))

		text_list_opinion, target_opinion = self.list_article_text('./training_set/foxnews_opinion.txt', 1)
		text_list_opinion = list(set(text_list_opinion))
		target_opinion = list(np.ones(len(text_list_opinion)))
		# text_list_opinion, target_opinion = strip_stopwords(text_list_opinion, target_opinion)

		self.text_list_news_dom, self.target_news_dom, self.text_list_news_world, \
			self.target_news_world, self.text_list_news, self.target_news, self.text_list_opinion, self.target_opinion =\
			text_list_news_dom, target_news_dom, text_list_news_world, \
			target_news_world, text_list_news, target_news, text_list_opinion, target_opinion
		if pickle_flag:
			with open('data_mnb.pkl', 'wb') as p:
				pickle.dump([text_list_news_dom, target_news_dom, text_list_news_world, \
					target_news_world, text_list_news, target_news, text_list_opinion, target_opinion], p)

	def load_with_pickle(self):
		with open('data_mnb.pkl') as p:
			self.text_list_news_dom, self.target_news_dom, \
			self.text_list_news_world, self.target_news_world, \
			self.text_list_news, self.target_news, \
			self.text_list_opinion, self.target_opinion =pickle.load(p)

	def train(self):
		news_zip = zip(self.text_list_news, self.target_news)
		opinion_zip = zip(self.text_list_opinion, self.target_opinion)
		shuffle(news_zip)
		shuffle(opinion_zip)
		text_list_news = [x[0] for x in news_zip]; target_news=[x[1] for x in news_zip]
		text_list_news_long = text_list_news
		text_list_opinion = [x[0] for x in opinion_zip]; target_opinion=[x[1] for x in opinion_zip]
		text_list_opinion_long = text_list_opinion

		limit_texts = min([len(text_list_news), len(text_list_opinion)])
		text_list_news, target_news = text_list_news[:limit_texts], target_news[:limit_texts]
		text_list_opinion, target_opinion = text_list_opinion[:limit_texts], target_opinion[:limit_texts]

		test_proportion = .75 #misnomer. it's actually train proportion but i'm too lazy to change the name in all these statements
		train_text_list_news, train_target_news = text_list_news[:int(len(target_news)*test_proportion)], target_news[:int(len(target_news)*test_proportion)]
		train_text_list_opinion, train_target_opinion = text_list_opinion[:int(len(target_opinion)*test_proportion)], target_opinion[:int(len(target_opinion)*test_proportion)]
		train_text = train_text_list_news+train_text_list_opinion
		train_target = train_target_news+train_target_opinion

		test_text_list_news, test_target_news = text_list_news[int(len(target_news)*test_proportion):], target_news[int(len(target_news)*test_proportion):]
		test_text_list_opinion, test_target_opinion = text_list_opinion[int(len(target_opinion)*test_proportion):], target_opinion[int(len(target_opinion)*test_proportion):]
		test_text = test_text_list_news+test_text_list_opinion
		test_target = test_target_news+test_target_opinion

		self.vectorizer = TfidfVectorizer(encoding='latin1')

		##This is new an untested
		count_vectorizer = CountVectorizer(encoding='latin1', min_df=1)
		transformer = TfidfTransformer()

		C_train = count_vectorizer.fit_transform((text for text in train_text))
		tfidf = transformer.fit_transform(C_train.toarray())
		weights=transformer.idf_ 
		###############
		X_train = self.vectorizer.fit_transform((text for text in train_text))
		assert sp.issparse(X_train)
		y_train = train_target 

		X_test = self.vectorizer.transform((text for text in test_text))
		y_test = test_target

		parameters={'alpha': 0.01}
		self.clf = MultinomialNB(**parameters).fit(X_train, y_train)

	###############################################################################
	# Benchmark classifiers
	def benchmark(self, clf_class=MultinomialNB, params={'alpha': 0.01}, name='MultinomialNB'):
	    print("parameters:", params)
	    t0 = time()
	    clf = clf_class(**params).fit(X_train, y_train)
	    print("done in %fs" % (time() - t0))

	    if hasattr(clf, 'coef_'):
	        print("Percentage of non zeros coef: %f"
	              % (np.mean(clf.coef_ != 0) * 100))
	    print("Predicting the outcomes of the testing set")
	    t0 = time()
	    pred = clf.predict(X_test)
	    print("done in %fs" % (time() - t0))

	    print("Classification report on test set for classifier:")
	    print(clf)
	    print()
	    print(classification_report(y_test, pred,
	                                target_names=['news','opinion']))

	    cm = confusion_matrix(y_test, pred)
	    print("Confusion matrix:")
	    print(cm)


	def predict(self, raw_text=None, fname=None):
		if fname:
			f = open(fname, 'r')
			a_text=[]
			target=[]
			for line in f:
				a_text.append(line)
			f.close()
			a_text = ' '.join(a_text)
		if raw_text:
			a_text = raw_text

		vect =  self.vectorizer.transform([a_text])
		pred = self.clf.predict(vect)
		pred_prob=self.clf.predict_proba(vect)
		return pred, pred_prob


##  What words make the certain article that way?  From words, select the top 10 most impactful words.