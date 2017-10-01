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
import signal


TEST_FRACTION = 0.1


def parse_args():
	parser = argparse.ArgumentParser(description="Train recurrent network")
	parser.add_argument("path", help="Text corpus")
	parser.add_argument("-l", "--limit", type=int, help="Maximum number of bytes to load from dataset")
	parser.add_argument("-e", "--epochs", type=int, default=1, help="Maximum number of bytes to load from dataset")
	parser.add_argument("-b", "--batch-size", type=int, default=128, dest="batch_size", help="Maximum number of bytes to load from dataset")
	return parser.parse_args()


def load_data(path, limit=None):
	infile = fileinput.FileInput(path, mode="r", openhook=fileinput.hook_compressed)
	infile = map(str, infile) # fileinput seems to ignore mode and always read binary?
	reader = csv.DictReader(infile, delimiter=";")

	power_consumption = []
	for row in reader:
		try:
			power_consumption.append(float(row["Global_active_power"]))
		except ValueError:
			pass # missing value
		if limit is not None and len(power_consumption) >= limit:
			break

	return np.asarray(power_consumption)


def normalize(data):
	mean = np.mean(data)
	data = data - mean
	return data, mean


def create_chunks(data, length, offset):
	chunks = more_itertools.windowed(data, length, step=offset)
	chunks = more_itertools.rstrip(chunks, lambda x: None in x) # Skip last chunk if it is incomplete

	return np.asarray(list(chunks))


def split_train_test(data, test_fraction):
	split_idx = round(data.shape[0] * (1-test_fraction))
	train = data[:split_idx]
	test = data[split_idx:]

	np.random.shuffle(train)

	X_train, y_train, X_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]

	return X_train, y_train, X_test, y_test


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


def run(model, epochs, batch_size, X_train, y_train, X_test):
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
	preds = model.predict(X_test)
	preds = np.squeeze(preds)
	return preds


def plot_results(y_test, pred):
	num_points = 100
	plt.plot(y_test[:num_points], label="y true")
	plt.plot(pred[:num_points], label="y pred")
	plt.legend(loc="upper left")
	plt.show()


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	data = load_data(args.path, args.limit)
	data, mean = normalize(data)
	data = create_chunks(data, 50, 1)

	X_train, y_train, X_test, y_test = split_train_test(data, TEST_FRACTION)
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

	model = build_model()

	print("X_train.shape:\t{0}".format(X_train.shape))
	print("y_train.shape:\t{0}".format(y_train.shape))
	print("X_test.shape:\t{0}".format(X_test.shape))
	print("y_test.shape:\t{0}".format(y_test.shape))
	model.summary()

	preds = run(model, args.epochs, args.batch_size, X_train, y_train, X_test)

	plot_results(y_test, preds)