from app import app
from flask import render_template
import MySQLdb as mdb
import os
import urllib2
import cookielib
import re
import time
from BeautifulSoup import BeautifulSoup
import goose
import mnb_live
from flask import request
import judge_url
import numpy as np
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# model = mnb_live.Model()
# model.reload_raw_data()
# train_text, test_text, train_target, test_target = model.prepare_train_and_test_sets()
# model.train()

# with open('trained_objects.pkl') as f:
#     m_list = pickle.load(f)
# model = m_list[0]


# def get_top_news_urls(BASE = 'http://www.nytimes.com', limit = 5):
#     def request_url(url, txdata, txheaders):
#         """Gets a webpage's HTML."""
#         req = Request(url, txdata, txheaders)
#         handle = urlopen(req)
#         html = handle.read()
#         return html

#     def remove_html_tags(data):
#         """Removes HTML tags"""
#         p = re.compile(r'< .*?>')
#         return p.sub('', data)

#     URL_REQUEST_DELAY = 1
#     TXDATA = None
#     TXHEADERS = {'User-agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
#     urlopen = urllib2.urlopen
#     Request = urllib2.Request

#     # Install cookie jar in opener for fetching URL
#     cookiejar = cookielib.LWPCookieJar()
#     opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookiejar))
#     urllib2.install_opener(opener)
#     html = request_url('http://www.nytimes.com/', TXDATA, TXHEADERS)

#     # Use BeautifulSoup to easily navigate HTML tree
#     soup = BeautifulSoup(html)

#     # Retrieves html from each url on NYT Global homepage under "story" divs
#     # with h2, h3, or h5 headlines
#     urls = []
#     for story in soup.findAll('div', {'class': 'story'}):
#         for hTag in story.findAll({'h1': True, 'h5': True,'h6': True,'h3': True, },
#                                   recursive=False):
#         # for hTag in story.findAll():
#             if hTag.find('a') and hTag.find('a')['href'].startswith(BASE+'/2013'):
#                 urls.append(hTag.find('a')['href'])
#                 if len(urls)>=limit:    
#                     return urls

@app.route('/')
@app.route('/index')
def index():
    ## Train model    
    # model = mnb_live.Model()
    # model.reload_raw_data()
    # model.train()

    # urls = get_top_news_urls(BASE = 'http://www.nytimes.com', limit = 5)
    # articles = []
    # for url in urls:
    #     j  = judge_url.JudgeUrl(url)
    #     articles.append(j.a)
    #################################################### Examples
    example_urls = ['http://www.nytimes.com/2013/09/11/opinion/friedman-threaten-to-threaten.html?_r=0',
                        'http://www.nytimes.com/2013/09/17/world/europe/us-and-allies-tell-syria-to-dismantle-chemical-arms-quickly.html?ref=world'
                            ]

    # possible_examples = ['http://nypost.com/2013/09/15/obamacare-will-question-your-sex-life/',
    #                         'http://www.dailykos.com/story/2013/04/26/1204994/-Bush-will-go-down-in-history-as-person-who-doomed-the-planet',
    #                         ]
    examples = []
    for url in example_urls:
        j  = judge_url.JudgeUrl(url)
        examples.append(j.a)

    # but_fox = ['http://www.foxnews.com/politics/2013/09/13/emails-show-irs-official-lerner-involved-in-tea-party-screening/']
    # but_fox_examples = []
    # for url in but_fox:
    #     j  = judge_url.JudgeUrl(url)
    #     but_fox_examples.append(j.a)

    # they_all_do_it = ['http://www.washingtontimes.com/news/2013/sep/1/john-kerry-evidence-nerve-agent-sarin-syria/?page=all#pagebreak',
    #                     'http://www.nytimes.com/2013/09/13/us/politics/at-meeting-with-treasury-secretary-boehner-pressed-for-debt-ceiling-deal.html?ref=politics&pagewanted=all',
    #                     'http://online.wsj.com/article/SB10001424127887323846504579071514012606076.html?mod=WSJ_MIDDLESecondStories',
    #                     ]
    # all_do_it_examples = []
    # for url in they_all_do_it:
    #     j  = judge_url.JudgeUrl(url)
    #     all_do_it_examples.append(j.a)
    ####################################################  MySQL PART
    # con = mdb.connect('localhost', 'testuser', 'test623', 'testdb');

    # with con: 
    #     cur = con.cursor()
    #     cur.execute("SELECT * FROM Writers")
    #     rows = cur.fetchall()
    # writers = rows
    #########################################################3
    return render_template("index.html",
        title = 'Home',
        examples = examples,
        # but_fox_examples = but_fox_examples,
        # all_do_it_examples = all_do_it_examples,
        # articles = articles
        )

@app.route('/slides')
def slides():
    pass


@app.route('/contact')
def contact():
    pass


#################
## Bing search api, slightly edited from https://github.com/xthepoet/pyBingSearchAPI

import requests # Get from https://github.com/kennethreitz/requests
import string

class BingSearchAPI():
    bing_api =  "https://api.datamarket.azure.com/Bing/Search/"
    
    def __init__(self, key):
        self.key = key

    def replace_symbols(self, request):
        # Custom urlencoder.
        # They specifically want %27 as the quotation which is a single quote '
        # We're going to map both ' and " to %27 to make it more python-esque
        request = string.replace(request, "'", '%27')
        request = string.replace(request, '"', '%27')
        request = string.replace(request, '+', '%2b')
        request = string.replace(request, ' ', '%20')
        request = string.replace(request, ':', '%3a')
        return request
        
    def search(self, sources, query, params):
        ''' This function expects a dictionary of query parameters and values.
            Sources and Query are mandatory fields.
            Sources is required to be the first parameter.
            Both Sources and Query requires single quotes surrounding it.
            All parameters are case sensitive. Go figure.

            For the Bing Search API schema, go to http://www.bing.com/developers/
            Click on Bing Search API. Then download the Bing API Schema Guide
            (which is oddly a word document file...pretty lame for a web api doc)
            '''
        request = sources + '?$format=json'
        query = self.replace_symbols(str(query))
        request += '&Query=%27' + str(query) + '%27'
        for key,value in params.iteritems():
            request += '&' + key + '=' + str(value)
        request = self.bing_api + request
        print request
        return requests.get(request, auth=(self.key, self.key))



import markdown
from flask import Flask
from flask import render_template
from flask import Markup

def bold_words(original_text, words_to_bold):
    bolded = original_text
    for word in words_to_bold:
        if word[2]>1:
            bolded = re.sub('(?i)( '+word[3]+' )', r'**\1**', bolded)
    return bolded

def bold_sents(original_text, sent_to_bold):
    bolded = original_text
    for sent in sent_to_bold:
        if sent[0]>-9:
            if sent[1].startswith('"\n\n'):
                bolded = bolded.replace(sent[1].lstrip('"\n'), '**'+sent[1].lstrip('"\n')+'**')
            else:
                bolded = bolded.replace(sent[1], '**'+sent[1]+'**')
    return bolded



@app.route('/article')
def article():
    # model = mnb_live.Model()
    # model.reload_raw_data()
    # train_text, test_text, train_target, test_target = model.prepare_train_and_test_sets()
    # model.train()
    stime=time.time()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
    # url = request.args.get('url', '')
    url = request.args.get('request_url', None)
    response = opener.open(url)
    raw_html = response.read()
    g = goose.Goose()
    a = g.extract(raw_html=raw_html)
    a.html_text = Markup(markdown.markdown(a.cleaned_text))
    print 'scrape: '+str(time.time()-stime)

    # pred, pred_prob = model.predict(raw_text=a.cleaned_text)
    j = judge_url.JudgeUrl('')
    pred_prob = j.evaluate_raw_text(a.cleaned_text)
    a.score = int(pred_prob[0][1]*100.)

    # stime=time.time()
    # op_words = model.why_opinion_faster()
    # print 'why: '+str(time.time()-stime)    
    # bolded_text = bold_words(a.cleaned_text, op_words)
    # a.html_text = Markup(markdown.markdown(bolded_text))


    # stime=time.time()
    # my_key = "vknjCZkZel4gofUWhubpLS0pXUXLbD5VqzIFgkXUHCg="
    # query_string = '"'+j.search_string+'"'
    # bing = BingSearchAPI(my_key)
    # params = {
    #         # 'ImageFilters':'"Face:Face"',
    #         #   '$format': 'json',
    #         #   '$top': 10,
    #         #   '$skip': 0
    #           }
    # alt_articles=[]       
    # bing_results = bing.search('Web',query_string,params).json() # requests 1.0+
    # results = bing_results['d']['results']
    # for i,result in enumerate(results):
    #     if i >= 3:
    #         break
    #     opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
    #     url_alt = result['Url']
    #     url_dom = url_alt.split('/')[2]
    #     response = opener.open(url_alt)
    #     raw_html = response.read()
    #     g = goose.Goose()
    #     art = g.extract(raw_html=raw_html)
    #     pred, pred_prob = model.predict(raw_text=art.cleaned_text)
    #     alt_articles.append({'url':url_alt, 'score':int(pred_prob[0][1]*100), 'source':url_dom})
    # print 'bing and predict: '+str(time.time()-stime)
    with open('temp_cleaned_text.pkl', 'w') as f:
        pickle.dump([a.cleaned_text, a.title, j.search_string],f)

    return render_template("article.html",
        url=url,
        a=a,
        # alt_articles = alt_articles, 
        main_text = a.html_text
        )

@app.route('/store_alternatives')
def store_alternatives():
    with open('temp_cleaned_text.pkl', 'r') as f:
        [cleaned_text, title, search_string] = pickle.load(f)

    my_key = "vknjCZkZel4gofUWhubpLS0pXUXLbD5VqzIFgkXUHCg="
    query_string = '"'+search_string+'"'
    print search_string
    bing = BingSearchAPI(my_key)
    params = {
              }
    alt_articles=[]       
    bing_results = bing.search('News',query_string,params).json() # requests 1.0+
    results = bing_results['d']['results']
    for i,result in enumerate(results):
        if i >= 5:
            break
        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
        url_alt = result['Url']
        url_dom = url_alt.split('/')[2]
        response = opener.open(url_alt)
        raw_html = response.read()
        g = goose.Goose()
        art = g.extract(raw_html=raw_html)
        # pred, pred_prob = model.predict(raw_text=art.cleaned_text)
        j = judge_url.JudgeUrl('')
        pred_prob = j.evaluate_raw_text(art.cleaned_text)
        alt_articles.append({'url':url_alt, 'score':int(pred_prob[0][1]*100), 'source':url_dom})
    alt_articles=sorted(alt_articles, key = lambda x:x['score'])
    with open('alternatives.pkl', 'w') as f:
        pickle.dump(alt_articles,f)
    return True


@app.route('/store_why')
def store_why():
    print 'store why is starting'
    # op_words = model.why_opinion_faster()
    # op_sents = model.rank_sents()   
    with open('trained_objects.pkl', 'r') as p:
            [feature_log_prob, intercept, training_word_list, training_counts] = pickle.load(p)
    with open('temp_cleaned_text.pkl', 'r') as f:
        [cleaned_text, title, search_string] = pickle.load(f)
    tfidf_transformer = TfidfTransformer()
    tfidf_training = tfidf_transformer.fit_transform(training_counts.toarray())
    count_vectorizer_sample = CountVectorizer(encoding=u'utf-8', stop_words = nltk.corpus.stopwords.words('english'))


    choice1_logs = []
    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    sents =sent_tokenizer.tokenize( cleaned_text)
    for sent in sents:
        try:
            counts_sample = count_vectorizer_sample.fit_transform([sent]).toarray()[0]
        except ValueError:
            continue
        tokens_sample = count_vectorizer_sample.get_feature_names()
        counts_in_context = np.zeros(len(training_word_list)) #initialize zero array for all the words in document corpus
        for i, word in enumerate(tokens_sample):
            try:
                idx = training_word_list.index(word)
                counts_in_context[idx] = counts_sample[i]
            except ValueError:
                pass        
        ## Convert counts_in_context to tfidf
        tfidf_sample = tfidf_transformer.transform((counts_in_context)).toarray()[0]
        log_score = (sum(feature_log_prob[1]*tfidf_sample)+intercept)/sum(tfidf_sample)
        if log_score <-99:
            continue
        choice1_logs.append([log_score[0], sent])
    s_choice1_logs = sorted(choice1_logs, key = lambda x:x[0])
    s_choice1_logs.reverse()
    op_sents = s_choice1_logs

    print op_sents

    bolded_text = bold_sents(cleaned_text, op_sents)
    print bolded_text
    with open('bolded_text.pkl','w') as f:
        t=Markup(markdown.markdown(bolded_text))
        pickle.dump([t, op_sents],f)
    return True

from flask import jsonify
@app.route('/display_why')
def display_why():
    ## for words
    # with open('bolded_text.pkl', 'r') as f:
    #     t, op_words = pickle.load(f)
    #     op_words.reverse()
    # return jsonify({'main_text':t, 'op_words': op_words})

    ## for sentences
    with open('bolded_text.pkl', 'r') as f:
        t, op_sents = pickle.load(f)
        op_sents.reverse()
    return jsonify({'main_text':t, 'op_sents': op_sents})


@app.route('/display_alternatives')
def display_alternatives():
    with open('alternatives.pkl', 'r') as f:
        alt_articles = pickle.load(f)
    return jsonify({'aa':alt_articles})