#!/usr/bin/env python3

import argparse
import glob
import gzip
import html.parser
import json
import more_itertools
import numpy as np
import re
import signal


def parse_args():
	parser = argparse.ArgumentParser(description="Preprocess household power consumtion data set")
	parser.add_argument("-f", "--test-fraction", type=float, default=0.1, help="Portion of the data to use for the test set")
	parser.add_argument("-i", "--input-length", type=int, default=90, help="Input sequence length")
	parser.add_argument("-s", "--step", type=int, default=30, help="Offset from one chunk to the next")
	parser.add_argument("origin", help="The unprocessed data")
	parser.add_argument("train_path", metavar="train-path", help="The unprocessed data")
	parser.add_argument("test_path", metavar="test-path", help="The unprocessed data")
	return parser.parse_args()


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


def simplify_text(content):
	content = " ".join(content.split())
	content = re.sub(r"\[.*?\]", "", content) # Replace [...] (usually references like [1])
	content = re.sub(r"[^\w \.,:;!?|@#$%&()/_+\"-]", "", content, flags=re.ASCII) # Remove all but the listet characters
	content = content.lower()

	return content


def create_chunks(data, length, offset):
	chunks = more_itertools.windowed(data, length, step=offset)
	chunks = more_itertools.rstrip(chunks, lambda x: None in x) # Skip last chunk if it is incomplete

	return list(chunks)


def vectorize(chunks):
	alphabet = sorted(set().union(*chunks))
	char_to_idx = {c: i for i, c in enumerate(alphabet)}
	idx_to_char = {i: c for i, c in enumerate(alphabet)}

	data = np.zeros((len(chunks), len(chunks[0]), len(alphabet)), dtype=np.bool)
	for idx_chunk, chunk in enumerate(chunks):
		for idx_char, char in enumerate(chunk):
			data[idx_chunk, idx_char, char_to_idx[char]] = 1

	return data, idx_to_char


def split_train_test(data, test_fraction):
	split_idx = round(data.shape[0] * (1-test_fraction))
	train = data[:split_idx]
	test = data[split_idx:]

	X_train, y_train, X_test, y_test = train[:,:-1,:], train[:,-1:,:], test[:,:-1,:], test[:,-1:,:]
	return X_train, y_train, X_test, y_test


def save_data(path, X, y, idx_to_char):
	if not path.endswith(".npy.gz"):
		raise ValueError("Output path must end in .npy.gz")

	data = np.concatenate([X, y], axis=1)
	with gzip.GzipFile(path, "w") as outfile:
		np.save(outfile, data)

	codepath = path.replace(".npy.gz", ".json")
	with open(codepath, "w") as outfile:
		json.dump(idx_to_char, outfile, sort_keys=True)


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	parser = PG_HTML_parser()
	for inpath in sorted(glob.glob(args.origin + "/*.html")):
		with open(inpath) as infile:
			html = infile.read()
		parser.feed(html)

	data = parser.content
	data = simplify_text(data)

	chunks = create_chunks(data, args.input_length+1, args.step)

	data, idx_to_char = vectorize(chunks)
	X_train, y_train, X_test, y_test = split_train_test(data, args.test_fraction)

	save_data(args.train_path, X_train, y_train, idx_to_char)
	save_data(args.test_path, X_test, y_test, idx_to_char)