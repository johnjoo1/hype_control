from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import pylab as pl
import nltk

from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB



def list_article_text(fname, label):
	
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

def strip_stopwords(text_list, targets):
	idx_empty = [i for i,x in enumerate(text_list) if x=='']
	text_list[:] = [text for i,text in enumerate(text_list) if i not in idx_empty]
	targets[:] = [target for i,target in enumerate(targets) if i not in idx_empty]
	clean_list = []

	for text in text_list:
		tokens = nltk.word_tokenize(text)
		words = [w.lower() for w in tokens]
		filtered_words = [w for w in words if not w in nltk.corpus.stopwords.words('english')]
		clean_list.append(' '.join(filtered_words))
	return clean_list, targets

# def merge_txt_files(text_files):



text_list_news, target_news = list_article_text('reuters_top_stories.txt', 0)
print len(text_list_news)
text_list_news = list(set(text_list_news))
target_news = list(np.zeros(len(text_list_news)))
text_list_news, target_news = strip_stopwords(text_list_news, target_news)
print len(text_list_news)
text_list_opinion, target_opinion = list_article_text('foxnews_opinion.txt', 1)
print len(text_list_opinion)
text_list_opinion = list(set(text_list_opinion))
target_opinion = list(np.ones(len(text_list_opinion)))
text_list_opinion, target_opinion = strip_stopwords(text_list_opinion, target_opinion)
print len(text_list_opinion)

text_list_news, target_news = text_list_news[:178], target_news[:178]
text_list_opinion, target_opinion = text_list_opinion[:178], target_opinion[:178]

test_proportion = .1
train_text_list_news, train_target_news = text_list_news[:int(len(target_news)*test_proportion)], target_news[:int(len(target_news)*test_proportion)]
train_text_list_opinion, train_target_opinion = text_list_opinion[:int(len(target_opinion)*test_proportion)], target_opinion[:int(len(target_opinion)*test_proportion)]
train_text = train_text_list_news+train_text_list_opinion
train_target = train_target_news+train_target_opinion

test_text_list_news, test_target_news = text_list_news[int(len(target_news)*test_proportion):], target_news[int(len(target_news)*test_proportion):]
test_text_list_opinion, test_target_opinion = text_list_opinion[int(len(target_opinion)*test_proportion):], target_opinion[int(len(target_opinion)*test_proportion):]
test_text = test_text_list_news+test_text_list_opinion
test_target = test_target_news+test_target_opinion

vectorizer = TfidfVectorizer(encoding='latin1')
# vectorizer = CountVectorizer(encoding='latin1')
X_train = vectorizer.fit_transform((text for text in train_text))
assert sp.issparse(X_train)
y_train = train_target 

X_test = vectorizer.transform((text for text in test_text))
y_test = test_target

###############################################################################
# Benchmark classifiers
def benchmark(clf_class, params, name):
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

    # Show confusion matrix
    # pl.matshow(cm)
    # pl.title('Confusion matrix of the %s classifier' % name)
    # pl.colorbar()


print("Testbenching a linear classifier...")
parameters = {
    'loss': 'hinge',
    'penalty': 'l2',
    'n_iter': 50,
    'alpha': 0.00001,
    'fit_intercept': True,
}

benchmark(SGDClassifier, parameters, 'SGD')

print("Testbenching a MultinomialNB classifier...")
parameters = {'alpha': 0.01}

benchmark(MultinomialNB, parameters, 'MultinomialNB')

# pl.show()

###################
##  Prepare classifiers for use
###################
clf = MultinomialNB(**parameters).fit(X_train, y_train)
# clf = SGDClassifier(**parameters).fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob=clf.predict_proba(X_test)

## Some analysis
#####
feature_names = vectorizer.get_feature_names()
top10_news = np.argsort(clf.feature_log_prob_[0])[-10:]
top10_opinion = np.argsort(clf.feature_log_prob_[1])[-10:]
for i in top10_news:
	print str(i) + str(vectorizer.get_feature_names()[i]) + str(clf.feature_count_[0][i])#/clf.feature_count_[1][i])
print '\n'
for i in top10_opinion:
	print str(i) + str(vectorizer.get_feature_names()[i]) + str(clf.feature_count_[1][i])#/clf.feature_count_[0][i])
# important_features = [vectorizer.get_feature_names()[x] for x in idx_high_coefs[0]]


fname='test_article.txt'
def predict(fname):
	
	f = open(fname, 'r')
	a_text=[]
	target=[]
	for line in f:
		a_text.append(line)
	f.close()
	a_text = ' '.join(a_text)

	vect =  vectorizer.transform([a_text])
	pred = clf.predict(vect)
	pred_prob=clf.predict_proba(vect)
	print pred, pred_prob
