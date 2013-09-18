import mnb_live
import goose
import urllib2
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import numpy as np



class JudgeUrl(object):
	def __init__(self, url):
		# self.train_model()
		# self.get_article(url)
		# self.evaluate_article()
		with open('trained_objects.pkl', 'r') as p:
			[self.feature_log_prob, self.intercept, self.training_word_list, self.training_counts] = pickle.load(p)
		self.tfidf_transformer = TfidfTransformer()
		self.tfidf_training = self.tfidf_transformer.fit_transform(self.training_counts.toarray())
		if not url =='':
			self.get_article(url)
			self.evaluate_article()

	def train_model(self):
		self.model = mnb_live.Model()
		self.model.reload_raw_data()
		self.model.train()

	def get_article(self, url):
		if url =='':
			return 1
		opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
		response = opener.open(url)
		raw_html = response.read()
		g = goose.Goose()
		self.a = g.extract(raw_html=raw_html)
		self.a.url = url
		self.a.url_dom = url.split('/')[2]

	def evaluate_raw_text(self,raw_text):
		count_vectorizer_sample = CountVectorizer(encoding=u'utf-8', stop_words = nltk.corpus.stopwords.words('english'))
		counts_sample = count_vectorizer_sample.fit_transform([raw_text]).toarray()[0]
		tokens_sample = count_vectorizer_sample.get_feature_names()
		counts_in_context = np.zeros(len(self.training_word_list)) #initialize zero array for all the words in document corpus
		for i, word in enumerate(tokens_sample):
			try:
				idx = self.training_word_list.index(word)
				counts_in_context[idx] = counts_sample[i]
			except ValueError:
				pass		

		## Convert counts_in_context to tfidf
		tfidf_sample = self.tfidf_transformer.transform((counts_in_context)).toarray()[0]
		## Calculate prediction probability
		choice0 = sum(self.feature_log_prob[0]*tfidf_sample)+self.intercept
		choice1 = sum(self.feature_log_prob[1]*tfidf_sample)+self.intercept
		pred_prob = [[np.exp(choice0)/(np.exp(choice0)+np.exp(choice1)), np.exp(choice1)/(np.exp(choice0)+np.exp(choice1))]]

		##  Top words from tf-idf
		word_i = np.argsort(tfidf_sample)[-5:]
		top_words = [self.training_word_list[i].encode("utf8","ignore") for i in word_i]
		search_string = ''
		top_words.reverse()
		for word in top_words:
			search_string+=word+ ' '
		self.search_string = search_string
		
		return pred_prob

	def evaluate_article(self):
		count_vectorizer_sample = CountVectorizer(encoding=u'utf-8', stop_words = nltk.corpus.stopwords.words('english'))
		counts_sample = count_vectorizer_sample.fit_transform([self.a.cleaned_text]).toarray()[0]
		tokens_sample = count_vectorizer_sample.get_feature_names()
		counts_in_context = np.zeros(len(self.training_word_list)) #initialize zero array for all the words in document corpus
		for i, word in enumerate(tokens_sample):
			try:
				idx = self.training_word_list.index(word)
				counts_in_context[idx] = counts_sample[i]
			except ValueError:
				pass

		## Convert counts_in_context to tfidf
		tfidf_sample = self.tfidf_transformer.transform((counts_in_context)).toarray()[0]
		## Calculate prediction probability
		choice0 = sum(self.feature_log_prob[0]*tfidf_sample)+self.intercept
		choice1 = sum(self.feature_log_prob[1]*tfidf_sample)+self.intercept
		pred_prob = [[np.exp(choice0)/(np.exp(choice0)+np.exp(choice1)), np.exp(choice1)/(np.exp(choice0)+np.exp(choice1))]]

		# pred, pred_prob = self.model.predict(raw_text=self.a.cleaned_text)
		self.a.score = self.convert_prob2score(pred_prob)

		##  Top words from tf-idf
		word_i = np.argsort(tfidf_sample)[-5:]
		top_words = [self.training_word_list[i].encode("utf8","ignore") for i in word_i]
		search_string = ''
		top_words.reverse()
		for word in top_words:
			search_string+=word+ ' '
		self.search_string = search_string
		return self.a.score

	def convert_prob2score(self, pred_prob):
		return int(pred_prob[0][1]*100.)
