from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import pylab as pl
import nltk
import sys
import pickle
import time

from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from random import shuffle


import time
class Utilities(object):	
	def isTimeFormat(self,input):
	    try:
	        time.strptime(input, '%H:%M')
	        return True
	    except ValueError:
	        return False

class TextWrangle(object):
	def list_article_text(self, fname, target_num):
		f = open(fname, 'r')
		text_list = []
		text=[]
		util = Utilities()
		for line in f:
			if line.startswith('headline:'):
				text = ' '.join(text)
				text_list.append(text)
				text=[]
			elif line.startswith('byline:'):
				pass
			elif util.isTimeFormat(line):
				pass
			else:
				text.append(line)
		f.close()
		text_list = self.remove_duplicates(text_list)
		target = list(np.ones(len(text_list))*target_num)
		return text_list, target

	def merge_txt_lists(self, text_lists, same_proportion=True, randomize = True, shuffle_flag = True):
		merged = []
		min_num_texts = min([len(text) for text in text_lists])
		if same_proportion == True:
			for text_list in text_lists:
				merged += text_list[:min_num_texts]
		else:
			for text_list in text_lists:
				merged += text_list
		if shuffle_flag==True:
			shuffle(merged)
		else:
			pass
		return merged

	def remove_duplicates(self, text_list):
		return list(set(text_list))

class Model(object):

	def reload_raw_data(self, pickle_flag=False):
		tw = TextWrangle()
		## original training set
		# text_list_news_dom, target_news_dom = tw.list_article_text('./training_set/reuters_domestic_stories.txt',0)
		# text_list_news_world, target_news_world = tw.list_article_text('./training_set/reuters_world_stories.txt',0)

		## longer_training_set
		text_list_news_dom, target_news_dom = tw.list_article_text('./longer_training_set/reuters_domestic_stories.txt',0)
		text_list_news_world, target_news_world = tw.list_article_text('./longer_training_set/reuters_world_stories.txt',0)

		text_list_news = tw.merge_txt_lists([text_list_news_dom, text_list_news_world])
		target_news = list(np.zeros(len(text_list_news)))

		##original training set
		# text_list_opinion, target_opinion = tw.list_article_text('./training_set/foxnews_opinion.txt',1)

		##longer training set
		text_list_fox_opinion, target_fox_opinion = tw.list_article_text('./longer_training_set/foxnews_opinion.txt',1)
		text_list_nyt_opinion, target_nyt_opinion = tw.list_article_text('./longer_training_set/nytimes_opinion1.txt',1)
		self.text_list_fox_opinion = text_list_fox_opinion
		self.text_list_nyt_opinion = text_list_nyt_opinion

		text_list_opinion = tw.merge_txt_lists([text_list_fox_opinion, text_list_nyt_opinion])
		target_opinion = list(np.ones(len(text_list_opinion)))

		self.text_list_news_dom, self.target_news_dom, self.text_list_news_world, \
			self.target_news_world, self.text_list_news, self.target_news, self.text_list_opinion, self.target_opinion =\
			text_list_news_dom, target_news_dom, text_list_news_world, \
			target_news_world, text_list_news, target_news, text_list_opinion, target_opinion

	# 	if pickle_flag:
	# 		with open('data_mnb.pkl', 'wb') as p:
	# 			pickle.dump([text_list_news_dom, target_news_dom, text_list_news_world, \
	# 				target_news_world, text_list_news, target_news, text_list_opinion, target_opinion], p)
	# def load_with_pickle(self):
	# 	with open('data_mnb.pkl') as p:
	# 		self.text_list_news_dom, self.target_news_dom, \
	# 		self.text_list_news_world, self.target_news_world, \
	# 		self.text_list_news, self.target_news, \
	# 		self.text_list_opinion, self.target_opinion =pickle.load(p)
	def zip_shuffle_unzip(self, text_list, targets):
		text_zipped = zip(text_list, targets)
		shuffle(text_zipped)
		shuffled_text_list = [x[0] for x in text_zipped]
		shuffled_targets = [x[1] for x in text_zipped]
		return shuffled_text_list, shuffled_targets

	def equalize_text_lists(self, text_lists, targets_lists):
		'''
		Makes the lengths of the text lists equal.  Same for target lists.
		text_lists and targets_lists are lists of lists.
		text_lists = [[text1, text2, text3], [texta, textb, textc]]
		targets_lists = [[0,0,0], [1,1,1]]
		'''
		limit_size = min([len(text_list) for text_list in text_lists])
		equalized_text_lists = [text_list[:limit_size] for text_list in text_lists]
		equalized_targets_lists = [target_list[:limit_size] for target_list in targets_lists]
		return equalized_text_lists, equalized_targets_lists

	def separate_train_test_set(self, text_lists, targets_lists, train_proportion = .75):
		## MUST EQUALIZE LENGTH OF LISTS BEFORE THIS STEP
		num_train = int(len(text_lists[0])*train_proportion)

		train_text = []
		train_target = []
		test_text = []
		test_target = []
		for text_list in text_lists:
			train_text += text_list[:num_train]
			test_text += text_list[num_train:]
		for targets_list in targets_lists:
			train_target += targets_list[:num_train]
			test_target += targets_list[num_train:]
		return train_text, test_text, train_target, test_target

	def prepare_train_and_test_sets(self, shuffle_flag=True):
		if shuffle_flag:
			text_list_news, target_news = self.zip_shuffle_unzip(self.text_list_news, self.target_news)
			text_list_opinion, target_opinion = self.zip_shuffle_unzip(self.text_list_opinion, self.target_opinion)
		else:
			text_list_news, target_news = self.text_list_news, self.target_news
			text_list_opinion, target_opinion = self.text_list_opinion, self.target_opinion
		equalized_text_lists, equalized_targets_lists = self.equalize_text_lists([text_list_news, text_list_opinion], [target_news, target_opinion])
		[self.text_list_news, self.text_list_opinion] = equalized_text_lists
		[self.target_news, self.target_opinion] = equalized_targets_lists

		train_text, test_text, train_target, test_target = \
			self.separate_train_test_set(equalized_text_lists, equalized_targets_lists, train_proportion=.75)
		return train_text, test_text, train_target, test_target

	def train(self):
		train_text, test_text, train_target, test_target = self.prepare_train_and_test_sets()
		self.train_text, self.test_text, self.train_target, self.test_target = train_text, test_text, train_target, test_target

		self.vectorizer = TfidfVectorizer(encoding=u'utf-8', stop_words = nltk.corpus.stopwords.words('english'))
		# self.vectorizer = CountVectorizer(encoding='latin1', stop_words = nltk.corpus.stopwords.words('english'))

		##This is new an untested
		# count_vectorizer = CountVectorizer(encoding='latin1', min_df=1)
		# transformer = TfidfTransformer()

		# C_train = count_vectorizer.fit_transform((text for text in train_text))
		# tfidf = transformer.fit_transform(C_train.toarray())
		# self.weights=transformer.idf_ 
		###############
		self.X_train = self.vectorizer.fit_transform((text for text in train_text))
		assert sp.issparse(self.X_train)

		# self.count_vectorizer.fit_transform((text for text in train_text))
		self.y_train = train_target 

		self.X_test = self.vectorizer.transform((text for text in test_text))
		self.y_test = test_target

		parameters={'alpha': 0.01}
		self.clf = MultinomialNB(**parameters).fit(self.X_train, self.y_train)

	def benchmark(self, clf_class=MultinomialNB, params={'alpha': 0.01}, name='MultinomialNB'):
	    print("parameters:", params)
	    t0 = time.time()
	    clf = clf_class(**params).fit(self.X_train, self.y_train)
	    print("done in %fs" % (time.time() - t0))

	    if hasattr(clf, 'coef_'):
	        print("Percentage of non zeros coef: %f"
	              % (np.mean(clf.coef_ != 0) * 100))
	    print("Predicting the outcomes of the testing set")
	    t0 = time.time()
	    pred = clf.predict(self.X_test)
	    print("done in %fs" % (time.time() - t0))

	    print("Classification report on test set for classifier:")
	    print(clf)
	    print()
	    print(classification_report(self.y_test, pred,
	                                target_names=['news','opinion']))

	    cm = confusion_matrix(self.y_test, pred)
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
			assert len(a_text)>0
		self.a_text = a_text
		vect =  self.vectorizer.transform([a_text])
		self.vect = vect
		pred = self.clf.predict(vect)
		pred_prob=self.clf.predict_proba(vect)
		return pred, pred_prob

	def why_opinion(self):
		feature_names = self.vectorizer.get_feature_names()
		opinion_idx = list(np.argsort(self.clf.feature_log_prob_[1])) #most important are last
		opinion_idx.reverse() #most important are first
		self.opinion_idx = opinion_idx
		i_filtered=0
		i_opinion_idx = 0
		filtered_keywords = []
		stime=time.time()
		while i_filtered<100:
			if not self.vectorizer.get_feature_names()[opinion_idx[i_opinion_idx]] in nltk.corpus.stopwords.words('english'):
				filtered_keywords.append(self.vectorizer.get_feature_names()[opinion_idx[i_opinion_idx]])
				i_filtered+=1
			i_opinion_idx+=1
		self.fk=filtered_keywords
		print 'why while: '+str(time.time()-stime)

		news_words=[]
		opinion_words = []
		ratio = self.clf.feature_log_prob_[1]-self.clf.feature_log_prob_[0]
		for i,count in enumerate(self.vect.toarray()[0]):
			if count>0:
				opinion_words.append([count*np.exp(ratio[i]), count, ratio[i], self.vectorizer.get_feature_names()[i]])
		opinion_words_sorted = sorted(opinion_words, key=lambda x:x[2])
		return opinion_words_sorted #, news_words_sorted

	def why_opinion_faster(self):
		opinion_words = []
		ratio = np.exp(self.clf.feature_log_prob_[1]-self.clf.feature_log_prob_[0])
		opinion_words = [ ( 0, #round(count*np.exp(ratio[i]),2) ,
							round(count,2) , 
							round(ratio[i],0) , 
							self.vectorizer.get_feature_names()[i] ) for i,count in enumerate(self.vect.toarray()[0]) if count>0]
		opinion_words_sorted = sorted(opinion_words, key=lambda x:x[2])
		opinion_words_sorted=[word for word in opinion_words_sorted if word[2]>1]
		# print opinion_words_sorted
		return opinion_words_sorted #, news_words_sorted




if __name__ == "__main__":
	m=Model()
	m.reload_raw_data()
	train_text, test_text, train_target, test_target = m.prepare_train_and_test_sets()
	m.train()

	####  Store important results from the training  ##########################
	# count_vectorizer = CountVectorizer(encoding=u'utf-8', stop_words = nltk.corpus.stopwords.words('english'))
	# training_counts = count_vectorizer.fit_transform((text for text in m.train_text))  ## store this!
	# assert(tfidf_vectorizer.get_feature_names() == tfidf_vectorizer.get_feature_names() ) #make sure word order is same
	# training_word_list = count_vectorizer.get_feature_names()  ## store this!

	# with open('trained_objects.pkl', 'wb') as p:
	# 	pickle.dump([m.clf.feature_log_prob_, m.clf.intercept_, training_word_list, training_counts], p)
	############################################################################

	pred, pred_prob = m.predict(fname="./sample_texts/news/fox_news.txt")
	# opinion_words, news_words = m.why_opinion()
	opinion_words = m.why_opinion_faster()
##  What words make the certain article that way?  From words, select the top 10 most impactful words.