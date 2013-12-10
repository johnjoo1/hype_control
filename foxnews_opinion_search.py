
##http://www.davidxia.com/2010/11/new-york-times-python-web-scraper/

## USE THIS FILE, NOT FOXNEWS_OPINION.PY
import os
import urllib2
import cookielib
import re
import time
from BeautifulSoup import BeautifulSoup
from HTMLParser import HTMLParser

URL_REQUEST_DELAY = 1
BASE = 'http://www.foxnews.com'
TXDATA = None
TXHEADERS = {'User-agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
OUTPUT_FILE = 'foxnews_opinion.txt'

def request_url(url, txdata, txheaders):
    """Gets a webpage's HTML."""
    req = Request(url, txdata, txheaders)
    handle = urlopen(req)
    html = handle.read()
    return html

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def remove_html_tags(data):
    """Removes HTML tags"""
    # p = re.compile(r'< .*?>')  #the original
    # return p.sub('', data)
    #return re.sub('<[^<]+?>', '', data) #regex that works
    s = MLStripper()
    s.feed(data)
    return s.get_data()


urlopen = urllib2.urlopen
Request = urllib2.Request

# Install cookie jar in opener for fetching URL
cookiejar = cookielib.LWPCookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookiejar))
urllib2.install_opener(opener)



def get_articles(html):
    # Use BeautifulSoup to easily navigate HTML tree
    soup = BeautifulSoup(html)

    # Retrieves html from each url on NYT Global homepage under "story" divs
    # with h2, h3, or h5 headlines

    urls = []
    for story in soup.findAll('div',{'class':'ez-main'}): #we are not getting first and last of the list
        # for hTag in story.findAll('div',{'class':'ez-main'}, recursive=False):
        for hTag in story.findAll():
            if hTag.find('a'):
                urls.append(hTag.find('a')['href'])

    # # Removes URLs that aren't news articles.
    # # Create a copy of list b/c you can't modify a list while iterating over it.
    for url in urls[:]:
        if not url.startswith('http://www.foxnews.com/opinion'):
            urls.remove(url)

    # Extracts headline, byline, dateline and content; outputs to file
    # if os.path.exists(OUTPUT_FILE):
    #     os.remove(OUTPUT_FILE)
    output = open(OUTPUT_FILE, 'a' )

    urls = list(set(urls)) #remove duplicate urls

    for url in urls[:]:
        content = ''
        html = request_url(url, TXDATA, TXHEADERS)
        # html = unicode(html, 'utf-8')
        soup = BeautifulSoup(html)
    #     # Gets HTML from single page link if article is > 1 page
    #     # if soup.find('li', {'class': 'singlePage'}):
    #     #     single = soup.find('li', {'class': 'singlePage'})
    #     #     html = request_url(BASE + single.find('a')['href'], TXDATA, TXHEADERS)
    #     #     html = unicode(html, 'utf-8')
    #     #     soup = BeautifulSoup(html)
    #     if 'singlepage' in html:
    #         if '?' in url:  
    #             html = request_url(url+'&pagewanted=all', TXDATA, TXHEADERS)
    #         else:
    #             html = request_url(url+'?pagewanted=all', TXDATA, TXHEADERS)
    #     html = unicode(html, 'utf-8')
    #     soup = BeautifulSoup(html)

        if not soup.find('h1'):
            continue
        headline = remove_html_tags(soup.find('h1').renderContents())  ##wtf? why isn't this working for <b>?
        print headline
        output.write('headline: '+headline + "\n")

        if not soup.find('p', {'itemprop':'author'}):
            continue
        byline = soup.find('p', {'itemprop':'author'}).renderContents()
        byline = remove_html_tags(byline) 
        output.write('byline: '+byline + "\n")

    #     dateline = soup.find('span', {'class': 'timestamp'}).renderContents()
    #     output.write(dateline)

        # article_section = soup.find('span', {'id':'articleText'})
        for p in soup.findAll('p'):
            # # Removes potential ad at the bottom of the page.
            # if p.findParents('div', {'class': 'ad'}):
            #     continue
            # # Prevents contents of nested <p> tags from being printed twice.
            # if p.findParents('div', {'class': 'authorIdentification'}):
            #     continue
            if '<div class="article-text" itemprop="articleBody">' in str(p.parent):
                if p.nextSibling==None:
                    pass
                elif p.renderContents()[0]=='(' and p.renderContents()[-1]==')':
                    pass
                else:
                    content = content + "\n\n" + p.renderContents().strip()
        content = remove_html_tags(content)
        content = re.sub(" +", " ", content)
        # content = unescape(unicode(content, 'utf-8'))
        content = content + "\n\n\n\n"
        output.write(content)

        time.sleep(URL_REQUEST_DELAY)

    output.close()

## USE THIS FILE, NOT FOXNEWS_OPINION.PY
for i in range(300):
    html = request_url('http://www.foxnews.com/search-results/search?&mc_Text=1292440&q=a&mc_Blog=17196&mc_Video=270034&f1=Opinion&mediatype=Text&start={i}'.format(i=i*10), TXDATA, TXHEADERS)
    get_articles(html)