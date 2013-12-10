
##http://www.davidxia.com/2010/11/new-york-times-python-web-scraper/

import os
import urllib2
import cookielib
import re
import time
from BeautifulSoup import BeautifulSoup

URL_REQUEST_DELAY = 1
BASE = 'http://www.reuters.com'
TXDATA = None
TXHEADERS = {'User-agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
OUTPUT_FILE = 'reuters_world_stories.txt'

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
    for story in soup.findAll('div', {'class': 'feature'}):
        for hTag in story.findAll({'h2': True}, recursive=False):
        # for hTag in story.findAll():
            if hTag.find('a'):
                urls.append(hTag.find('a')['href'])

    # Removes URLs that aren't news articles.
    # Create a copy of list b/c you can't modify a list while iterating over it.
    for url in urls[:]:
        if not url.startswith('/article'):
            urls.remove(url)

    # Extracts headline, byline, dateline and content; outputs to file
    # if os.path.exists(OUTPUT_FILE):
    #     os.remove(OUTPUT_FILE)
    output = open(OUTPUT_FILE, 'a' )

    urls = list(set(urls)) #remove duplicate urls

    for url in urls[:]:
        content = ''
        html = request_url(BASE+url, TXDATA, TXHEADERS)
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
        headline = soup.find('h1').renderContents()
        if ':' in headline: #these tend to be opinions or analysis
            continue
        print headline
        output.write('headline: '+headline + "\n")

        if not soup.find('p', {'class':'byline'}):
            continue
        byline = soup.find('p', {'class':'byline'}).renderContents()
        byline = remove_html_tags(byline) 
        output.write('byline: '+byline + "\n")

        dateline = soup.find('span', {'class': 'timestamp'}).renderContents()
        output.write(dateline)

        for p in soup.findAll('p'):
            # # Removes potential ad at the bottom of the page.
            # if p.findParents('div', {'class': 'ad'}):
            #     continue
            # # Prevents contents of nested <p> tags from being printed twice.
            # if p.findParents('div', {'class': 'authorIdentification'}):
            #     continue
            if 'midArticle' in str(p.previousSibling):
                if 'Reuters' in p.renderContents():
                    content=content+"\n\n"+p.renderContents().split('-')[-1].strip()
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

# html = request_url('http://www.reuters.com/news/us', TXDATA, TXHEADERS)
# get_articles(html)
for i in range(142):
    html=request_url('http://www.reuters.com/news/archive/worldNews?view=page&page={i}&pageSize=10'.format(i=i), TXDATA, TXHEADERS)
    get_articles(html)