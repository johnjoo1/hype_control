#!/home/johnjoo/Desktop/insight/hype_control/env/bin/python
from app import app
app.run(debug = True)
from flaskext.markdown import Markdown
Markdown(app)
