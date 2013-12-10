from scrapy.item import Item, Field

class ArticleItem(Item):
    headline = Field()
    body = Field()
    authors = Field()

class NYTimesSpider(CrawlSpider):
	name = 'nytimes.com'
    allowed_domains = ['nytimes.com']
    start_urls = ['http://www.nytimes.com']
    rules = [Rule(SgmlLinkExtractor(allow=['/tor/\d+']), 'parse_torrent')]

    def parse_torrent(self, response):
        x = HtmlXPathSelector(response)

        torrent = TorrentItem()
        torrent['url'] = response.url
        torrent['name'] = x.select("//h1/text()").extract()
        torrent['description'] = x.select("//div[@id='description']").extract()
        torrent['size'] = x.select("//div[@id='info-left']/p[2]/text()[2]").extract()
        return torrent

# <p itemprop="articleBody">
//p[@itemprop='articleBody']

# <h1 class="articleHeadline" itemprop="headline">
# <nyt_headline type=" " version="1.0">McCain Urges Lawmakers to Back Obama’s Plan for Syria</nyt_headline>
# </h1>
//nyt_headline/text()


//nyt_byline/h6/span/span[@itemprop="name"]
# <nyt_byline>
# <h6 class="byline">
# By
# <span itemid="http://topics.nytimes.com/top/reference/timestopics/people/c/jackie_calmes/index.html" itemtype="http://schema.org/Person" itemscope="" itemprop="author creator">
# <a title="More Articles by JACKIE CALMES" rel="author" href="http://topics.nytimes.com/top/reference/timestopics/people/c/jackie_calmes/index.html">
# <span itemprop="name">JACKIE CALMES</span>
# </a>
# </span>
# ,
# <span itemid="http://topics.nytimes.com/top/reference/timestopics/people/g/michael_r_gordon/index.html" itemtype="http://schema.org/Person" itemscope="" itemprop="author creator">
# <a title="More Articles by MICHAEL R. GORDON" rel="author" href="http://topics.nytimes.com/top/reference/timestopics/people/g/michael_r_gordon/index.html">
# <span itemprop="name">MICHAEL R. GORDON</span>
# </a>
# </span>
# and
# <span itemid="http://topics.nytimes.com/top/reference/timestopics/people/s/eric_schmitt/index.html" itemtype="http://schema.org/Person" itemscope="" itemprop="author creator">
# <a title="More Articles by ERIC SCHMITT" rel="author" href="http://topics.nytimes.com/top/reference/timestopics/people/s/eric_schmitt/index.html">
# <span itemprop="name">ERIC SCHMITT</span>
# </a>
# </span>
# </h6>
# </nyt_byline>