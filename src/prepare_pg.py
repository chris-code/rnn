#!/usr/bin/env python3

import glob
import html.parser
import re
import signal

PG_ROOT = "/home/chrisbot/Projects/Geany-Workspace/RNN/data/pg_raw"

class PG_HTML_parser(html.parser.HTMLParser):
	def __init__(self):
		super(PG_HTML_parser, self).__init__()
		self.content = ""

	def handle_starttag(self, tag, attrs):
		if tag == "p":
			self.is_p = True

	def handle_data(self, data):
		try:
			if self.is_p:
				self.content += data + " "
		except AttributeError:
			pass

	def handle_endtag(self, tag):
		if tag == "p":
			self.is_p = False

def simplify_content(content):
	content = " ".join(content.split())

	content = re.sub(r"\[.*?\]", "", content)

	content = re.sub(r"[^\w \.,:;!?|@#$%&()/_+\"-]", "", content, flags=re.ASCII)

	content = content.lower()

	return content

if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)

	for inpath in sorted(glob.glob(PG_ROOT + "/*.html")):
		with open(inpath) as infile:
			html = infile.read()

		parser = PG_HTML_parser()
		parser.feed(html)

		content = parser.content
		content = simplify_content(content)

		print(content, end=" ")