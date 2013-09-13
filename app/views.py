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

model = mnb_live.Model()
model.reload_raw_data()
train_text, test_text, train_target, test_target = model.prepare_train_and_test_sets()
model.train()
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
    fox_news_new_urls = ['http://www.foxnews.com/politics/2013/09/13/playing-with-us-assad-piles-on-demands-amid-chemical-weapons-talks/',
                            'http://www.foxnews.com/weather/2013/09/13/3-dead-in-colorado-floods/',
                            'http://www.foxnews.com/politics/2013/09/13/feds-looking-into-clinton-2008-campaign-for-links-to-dc-corruption-case/'
                            ]
    examples = []
    for url in fox_news_new_urls:
        j  = judge_url.JudgeUrl(url)
        examples.append(j.a)

    but_fox = ['http://www.foxnews.com/politics/2013/09/13/emails-show-irs-official-lerner-involved-in-tea-party-screening/']
    but_fox_examples = []
    for url in but_fox:
        j  = judge_url.JudgeUrl(url)
        but_fox_examples.append(j.a)

    they_all_do_it = ['http://www.washingtontimes.com/news/2013/sep/1/john-kerry-evidence-nerve-agent-sarin-syria/?page=all#pagebreak',
                        'http://www.nytimes.com/2013/09/13/us/politics/at-meeting-with-treasury-secretary-boehner-pressed-for-debt-ceiling-deal.html?ref=politics&pagewanted=all',
                        'http://online.wsj.com/article/SB10001424127887323846504579071514012606076.html?mod=WSJ_MIDDLESecondStories',
                        ]
    all_do_it_examples = []
    for url in they_all_do_it:
        j  = judge_url.JudgeUrl(url)
        all_do_it_examples.append(j.a)
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
        but_fox_examples = but_fox_examples,
        all_do_it_examples = all_do_it_examples,
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

    pred, pred_prob = model.predict(raw_text=a.cleaned_text)
    a.score = int(pred_prob[0][1]*100.)

    # stime=time.time()
    # op_words = model.why_opinion_faster()
    # print 'why: '+str(time.time()-stime)    
    # bolded_text = bold_words(a.cleaned_text, op_words)
    # a.html_text = Markup(markdown.markdown(bolded_text))


    stime=time.time()
    my_key = "vknjCZkZel4gofUWhubpLS0pXUXLbD5VqzIFgkXUHCg="
    query_string = '"'+a.title+'"'
    bing = BingSearchAPI(my_key)
    params = {
            # 'ImageFilters':'"Face:Face"',
            #   '$format': 'json',
            #   '$top': 10,
            #   '$skip': 0
              }
    alt_articles=[]       
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
        pickle.dump([a.cleaned_text, a.title],f)

    return render_template("article.html",
        url=url,
        a=a,
        alt_articles = alt_articles, 
        main_text = a.html_text
        )

@app.route('/store_alternatives')
def store_alternatives():
    with open('temp_cleaned_text.pkl', 'r') as f:
        [cleaned_text, title] = pickle.load(f)

    my_key = "vknjCZkZel4gofUWhubpLS0pXUXLbD5VqzIFgkXUHCg="
    query_string = '"'+title+'"'
    bing = BingSearchAPI(my_key)
    params = {
              }
    alt_articles=[]       
    bing_results = bing.search('Web',query_string,params).json() # requests 1.0+
    results = bing_results['d']['results']
    for i,result in enumerate(results):
        if i >= 4:
            break
        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
        url_alt = result['Url']
        url_dom = url_alt.split('/')[2]
        response = opener.open(url_alt)
        raw_html = response.read()
        g = goose.Goose()
        art = g.extract(raw_html=raw_html)
        pred, pred_prob = model.predict(raw_text=art.cleaned_text)
        alt_articles.append({'url':url_alt, 'score':int(pred_prob[0][1]*100), 'source':url_dom})
    with open('alternatives.pkl', 'w') as f:
        pickle.dump(alt_articles,f)
    return True
 

@app.route('/store_why')
def store_why():
    print 'store why is starting'
    op_words = model.why_opinion_faster()   
    with open('temp_cleaned_text.pkl', 'r') as f:
        [cleaned_text, title] = pickle.load(f)
    bolded_text = bold_words(cleaned_text, op_words)
    with open('bolded_text.pkl','w') as f:
        t=Markup(markdown.markdown(bolded_text))
        pickle.dump([t, op_words],f)
    return True

from flask import jsonify
@app.route('/display_why')
def display_why():
    with open('bolded_text.pkl', 'r') as f:
        t, op_words = pickle.load(f)
        op_words.reverse()
    return jsonify({'main_text':t, 'op_words': op_words})


@app.route('/display_alternatives')
def display_alternatives():
    with open('alternatives.pkl', 'r') as f:
        alt_articles = pickle.load(f)
    return jsonify({'aa':alt_articles})