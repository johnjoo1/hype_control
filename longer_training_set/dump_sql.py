import MySQLdb as mdb
import time

def remove_duplicates(text_list):
	return list(set(text_list))

def isTimeFormat(input):
    try:
        time.strptime(input, '%H:%M')
        return True
    except ValueError:
        return False

def list_article_text( fname):
	f = open(fname, 'r')
	text_list = []
	text=[]
	for line in f:
		if line.startswith('headline:'):
			text = ' '.join(text)
			text_list.append(text)
			text=[]
		elif line.startswith('byline:'):
			pass
		elif isTimeFormat(line):
			pass
		else:
			text.append(line)
	f.close()
	text_list = remove_duplicates(text_list)
	return text_list

fnames = [['foxnews_opinion.txt', 'foxnews', 'opinion'],
	['reuters_domestic_stories.txt', 'reuters_domestic', 'news'],
	['reuters_world_stories.txt', 'reuters_world', 'news'],
	['nytimes_opinion1.txt', 'nytimes', 'opinion']]

data_list = []
for fname in fnames:
    text_list = list_article_text(fname[0])
    for article in text_list:
        data_list.append([fname[1], fname[2], article])

con = mdb.connect('localhost', 'root', '', 'training_set');


with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Articles")
    cur.execute("CREATE TABLE Articles(Id INT PRIMARY KEY AUTO_INCREMENT, \
                 Source ENUM('foxnews', 'nytimes', 'reuters_world', 'reuters_domestic'), \
                 Article_type ENUM('news', 'opinion'),\
                 article_TXT LONGTEXT CHARACTER SET utf8)")
    beg = 0
    end = 500
    while end<len(data_list):
        cur.executemany("INSERT INTO Articles(Source, Article_type, article_TXT) VALUES(%s, %s, %s)", data_list[beg:end])
        beg = end
        end += 500
