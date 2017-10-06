#!/usr/bin/env python3

import argparse
import gzip
import json
import keras
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import signal


def parse_args():
	parser = argparse.ArgumentParser(description="Train recurrent network")
	parser.add_argument("-l", "--limit", type=int, help="Maximum number of data points to load from dataset")
	parser.add_argument("-t", "--num-targets", type=int, dest="num_targets", default=1, help="Length of the target sequence")
	parser.add_argument("-b", "--batch-size", type=int, default=128, dest="batch_size", help="How many points comprise a batch update")
	parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train for")
	parser.add_argument("path", help="Training data")
	parser.add_argument("output_path", metavar="output-path", help="Where to store the model")
	return parser.parse_args()


def load_data(path, num_targets, limit=None):
	with gzip.GzipFile(path, "r") as infile:
		data = np.load(infile)
	data = data[:limit]

	X = data[:,:-num_targets,:]
	y = data[:,-num_targets:,:]

	codepath = path.replace(".npy.gz", ".json")
	with open(codepath, "r") as infile:
		idx_to_char = json.load(infile)
	idx_to_char = {int(k): v for k, v in idx_to_char.items()}

	return X, y, idx_to_char


def build_model(sequence_length, alphabet_size):
	optimizer = keras.optimizers.Adadelta(lr=0.1)

	model = keras.models.Sequential()
	model.add(LSTM(256, input_shape=(sequence_length, alphabet_size)))
	model.add(Dropout(0.5))
	model.add(Dense(alphabet_size, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=optimizer)

	return model


def save_model(path, model):
	model.save(path)


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

	X, y, idx_to_char = load_data(args.path, args.num_targets, args.limit)
	y = np.squeeze(y)
	print("X.shape:\t{0}".format(X.shape))
	print("y.shape:\t{0}".format(y.shape))

	model = build_model(X.shape[1], X.shape[2])
	model.summary()
	model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs)

	save_model(args.output_path, model)

	#seed_idx = random.randint(0, len(data)-SENTENCE_LENGTH-1)
	#seed = data[seed_idx:seed_idx+SENTENCE_LENGTH]
	#pred = seed
	#for i in range(PREDICT_LENGTH):
		#pred += predict(model, pred[-SENTENCE_LENGTH:], len(alphabet), char_to_idx)

	#print("Seed:\t" + seed)
	#print("Pred:\t" + pred)











