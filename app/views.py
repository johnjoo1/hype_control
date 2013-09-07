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


def get_top_news_urls(BASE = 'http://www.nytimes.com', limit = 5):
    def request_url(url, txdata, txheaders):
        """Gets a webpage's HTML."""
        req = Request(url, txdata, txheaders)
        handle = urlopen(req)
        html = handle.read()
        return html

    def remove_html_tags(data):
        """Removes HTML tags"""
        p = re.compile(r'< .*?>')
        return p.sub('', data)

    URL_REQUEST_DELAY = 1
    TXDATA = None
    TXHEADERS = {'User-agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
    urlopen = urllib2.urlopen
    Request = urllib2.Request

    # Install cookie jar in opener for fetching URL
    cookiejar = cookielib.LWPCookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookiejar))
    urllib2.install_opener(opener)
    html = request_url('http://www.nytimes.com/', TXDATA, TXHEADERS)

    # Use BeautifulSoup to easily navigate HTML tree
    soup = BeautifulSoup(html)

    # Retrieves html from each url on NYT Global homepage under "story" divs
    # with h2, h3, or h5 headlines
    urls = []
    for story in soup.findAll('div', {'class': 'story'}):
        for hTag in story.findAll({'h1': True, 'h5': True,'h6': True,'h3': True, },
                                  recursive=False):
        # for hTag in story.findAll():
            if hTag.find('a') and hTag.find('a')['href'].startswith(BASE+'/2013'):
                urls.append(hTag.find('a')['href'])
                if len(urls)>=limit:    
                    return urls

@app.route('/')
@app.route('/index')
def index():
    ## Train model    
    model = mnb_live.Model()
    model.reload_raw_data()
    model.train()

    urls = get_top_news_urls(BASE = 'http://www.nytimes.com', limit = 5)
    articles = []
    # opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
    for url in urls:
        j  = judge_url.JudgeUrl(url)
        articles.append(j.a)
        # response = opener.open(url)
        # raw_html = response.read()
        # g = goose.Goose()
        # a = g.extract(raw_html=raw_html)
        # pred, pred_prob = model.predict(raw_text=a.cleaned_text)
        # a.score = int(pred_prob[0][1]*100.)
        # a.url=url
        # articles.append(a)
    ####################################################  MySQL PART
    con = mdb.connect('localhost', 'testuser', 'test623', 'testdb');

    with con: 
        cur = con.cursor()
        cur.execute("SELECT * FROM Writers")
        rows = cur.fetchall()
    writers = rows
    #########################################################3
    return render_template("index.html",
        title = 'Home',
        articles = articles
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


@app.route('/article')
def article():
    model = mnb_live.Model()
    model.reload_raw_data()
    model.train()

    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
    url = request.args.get('url', '')
    response = opener.open(url)
    raw_html = response.read()
    g = goose.Goose()
    a = g.extract(raw_html=raw_html)
    pred, pred_prob = model.predict(raw_text=a.cleaned_text)
    a.score = int(pred_prob[0][1]*100.)

    my_key = "vknjCZkZel4gofUWhubpLS0pXUXLbD5VqzIFgkXUHCg="
    query_string = '"'+a.title+'"'
    bing = BingSearchAPI(my_key)
    params = {
            # 'ImageFilters':'"Face:Face"',
            #   '$format': 'json',
            #   '$top': 10,
            #   '$skip': 0
              }
    bing_results = bing.search('Web',query_string,params).json() # requests 1.0+
    results = bing_results['d']['results']
    alt_articles=[]
    for i,result in enumerate(results):
        if i >= 3:
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

    return render_template("article.html",
        url=url,
        a=a,
        alt_articles = alt_articles
        )