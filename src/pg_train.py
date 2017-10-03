#!/usr/bin/env python3

import argparse
import keras
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
import more_itertools
import numpy as np
import random
import signal

PG_PATH = "/home/chrisbot/Projects/Geany-Workspace/RNN/data/pg.txt"
SENTENCE_LENGTH = 90
PREDICT_LENGTH = 30


def parse_args():
	parser = argparse.ArgumentParser(description="Train recurrent network")
	parser.add_argument("path", help="Text corpus")
	parser.add_argument("-l", "--limit", type=int, help="Maximum number of bytes to load from dataset")
	parser.add_argument("-e", "--epochs", type=int, default=1, help="Maximum number of bytes to load from dataset")
	parser.add_argument("-b", "--batch-size", type=int, default=128, dest="batch_size", help="Maximum number of bytes to load from dataset")
	return parser.parse_args()


def load_data(path, limit=None):
	with open(path) as infile:
		return infile.read(limit)


def create_sentences(data, sentence_length):
	sentence_length += 1 # The last one will be the target
	chunks = more_itertools.windowed(data, sentence_length, step=sentence_length//3)
	chunks = more_itertools.rstrip(chunks, lambda x: None in x) # Skip last chunk because it may be too short

	sentences, next_chars = [], []
	for c in chunks:
		c = "".join(c)
		sentences.append(c[:-1])
		next_chars.append(c[-1])

	return sentences, next_chars


def vectorize(sentences, next_chars, alphabet):
	char_to_idx = {c: i for i, c in enumerate(alphabet)}
	idx_to_char = {i: c for i, c in enumerate(alphabet)}

	X = np.zeros((len(sentences), len(sentences[0]), len(alphabet)), dtype=np.bool)
	y = np.zeros((len(sentences), len(alphabet)), dtype=np.bool)

	for idx_s, sentence in enumerate(sentences):
		for idx_c, char in enumerate(sentence):
			X[idx_s, idx_c, char_to_idx[char]] = 1
		y[idx_s, char_to_idx[next_chars[idx_s]]] = 1

	return X, y, char_to_idx, idx_to_char


def build_model(sentence_length, alphabet_size):
	optimizer = keras.optimizers.Adadelta(lr=0.1)

	model = keras.models.Sequential()
	model.add(LSTM(256, input_shape=(sentence_length, alphabet_size)))
	model.add(Dropout(0.5))
	model.add(Dense(alphabet_size, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=optimizer)

	return model


def predict(model, seed, alphabet_size, char_to_idx):
	X = np.zeros((1, len(seed), alphabet_size))
	for idx, char in enumerate(seed):
		X[0, idx, char_to_idx[char]] = 1
	pred_probas = model.predict(X)[0]
	pred = idx_to_char[np.argmax(pred_probas)]
	return pred


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	data = load_data(args.path, args.limit)
	alphabet = sorted(set(data))
	sentences, next_chars = create_sentences(data, SENTENCE_LENGTH)
	X, y, char_to_idx, idx_to_char = vectorize(sentences, next_chars, alphabet)

	print("Corpus size:\t{0} chars".format(len(data)))
	print("Alphabet size:\t{0} ({1})".format(len(alphabet), "".join(alphabet)))
	print("Sentences:\t{0} (length {1})".format(len(sentences), SENTENCE_LENGTH))
	print("X.shape:\t{0}".format(X.shape))
	print("y.shape:\t{0}".format(y.shape))

	model = build_model(SENTENCE_LENGTH, len(alphabet))
	model.summary()
	model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs)

	seed_idx = random.randint(0, len(data)-SENTENCE_LENGTH-1)
	seed = data[seed_idx:seed_idx+SENTENCE_LENGTH]
	pred = seed
	for i in range(PREDICT_LENGTH):
		pred += predict(model, pred[-SENTENCE_LENGTH:], len(alphabet), char_to_idx)

	print("Seed:\t" + seed)
	print("Pred:\t" + pred)











