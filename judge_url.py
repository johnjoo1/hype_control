import mnb_live
import goose
import urllib2

class JudgeUrl(object):
	def __init__(self, url):
		self.train_model()
		self.get_article(url)
		self.evaluate_article()

	def train_model(self):
		self.model = mnb_live.Model()
		self.model.reload_raw_data()
		self.model.train()

	def get_article(self, url):
		opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
		response = opener.open(url)
		raw_html = response.read()
		g = goose.Goose()
		self.a = g.extract(raw_html=raw_html)
		self.a.url = url

	def evaluate_article(self):
		pred, pred_prob = self.model.predict(raw_text=self.a.cleaned_text)
		self.a.score = self.convert_prob2score(pred_prob)
		return self.a.score

	def convert_prob2score(self, pred_prob):
		return int(pred_prob[0][1]*100.)
