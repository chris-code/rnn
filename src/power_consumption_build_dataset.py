#!/usr/bin/env python3

import argparse
import csv
import fileinput
import more_itertools
import numpy as np
import signal


def parse_args():
	parser = argparse.ArgumentParser(description="Preprocess household power consumtion data set")
	parser.add_argument("-f", "--test-fraction", type=float, default=0.1, help="Portion of the data to use for the test set")
	parser.add_argument("-i", "--input-length", type=int, default=50, help="Input sequence length")
	parser.add_argument("-s", "--step", type=int, default=1, help="Offset from one chunk to the next")
	parser.add_argument("origin", help="The unprocessed data")
	parser.add_argument("train_path", metavar="train-path", help="The unprocessed data")
	parser.add_argument("test_path", metavar="test-path", help="The unprocessed data")
	return parser.parse_args()


def load_data(path, limit=None):
	infile = fileinput.FileInput(path, mode="r", openhook=fileinput.hook_compressed)
	infile = map(bytes.decode, infile) # fileinput seems to ignore mode and always read binary?
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


def save_data(path, X, y):
	header = ["t_{0}".format(i) for i in range(X.shape[1])]
	header.append("t_{0}_TARGET".format(X.shape[1]))
	data = np.hstack([X, y.reshape((-1, 1))])

	with open(path, "w") as outfile:
		writer = csv.writer(outfile, delimiter=str("\t"), lineterminator=str("\n"))
		writer.writerow(header)
		writer.writerows(data)


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	data = load_data(args.origin)
	chunks = create_chunks(data, args.input_length+1, args.step)
	X_train, y_train, X_test, y_test = split_train_test(chunks, args.test_fraction)

	save_data(args.train_path, X_train, y_train)
	save_data(args.test_path, X_test, y_test)