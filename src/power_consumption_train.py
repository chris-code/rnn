#!/usr/bin/env python3

import argparse
import csv
import fileinput
import keras
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import pickle
import signal
import sklearn.preprocessing


TEST_FRACTION = 0.1


def parse_args():
	parser = argparse.ArgumentParser(description="Train recurrent network")
	parser.add_argument("-l", "--limit", type=int, help="Maximum number of bytes to load from dataset")
	parser.add_argument("-e", "--epochs", type=int, default=1, help="Maximum number of bytes to load from dataset")
	parser.add_argument("-b", "--batch-size", type=int, default=128, dest="batch_size", help="Maximum number of bytes to load from dataset")
	parser.add_argument("data_path", metavar="data-path", help="The training data file")
	parser.add_argument("output_prefix", metavar="output_prefix", help="Prefix for storing the model")
	return parser.parse_args()


def load_data(path, limit=None):
	infile = fileinput.FileInput(path, openhook=fileinput.hook_compressed)
	infile = map(bytes.decode, infile) # fileinput seems to ignore mode and always read binary?
	reader = csv.DictReader(infile, delimiter=str("\t"))
	header = reader.fieldnames

	X_names = [h for h in header if not h.endswith("_TARGET")]
	y_names = [h for h in header if h.endswith("_TARGET")]
	target_indices = [header.index(t) for t in y_names]

	data = np.loadtxt(path, delimiter=str("\t"), skiprows=1)
	data = data[:limit]

	y = data[:,target_indices]
	X = np.delete(data, target_indices, axis=1)

	return X, y, X_names, y_names


def normalize(X, mean=None):
	if mean is None:
		mean = np.mean(X)
	X = X - mean
	return X, mean


def build_model():
	optimizer = keras.optimizers.Adadelta(lr=0.1)

	model = keras.models.Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(None, 1)))
	model.add(Dropout(0.2))
	model.add(LSTM(100, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation="linear"))
	model.compile(loss="mse", optimizer=optimizer)

	return model


def save_model(path, model, normalizer, X_names, y_names):
	model.save(path + ".model.keras")
	with open(path + ".model", "wb") as outfile:
		pickle.dump((normalizer, X_names, y_names), outfile)


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	X, y, X_names, y_names = load_data(args.data_path, args.limit)
	#X, mean = normalize(X)

	normalizer = sklearn.preprocessing.StandardScaler(with_std=False)
	X = normalizer.fit_transform(X)

	model = build_model()

	print("X.shape:\t{0}".format(X.shape))
	print("y.shape:\t{0}".format(y.shape))
	print("Inputs:\t" + "\t".join(X_names))
	print("Outputs:\t" + "\t".join(y_names))
	model.summary()

	X = np.reshape(X, (X.shape[0], X.shape[1], 1))
	model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size)

	save_model(args.output_prefix, model, normalizer, X_names, y_names)















