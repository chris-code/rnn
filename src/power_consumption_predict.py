#!/usr/bin/env python3

import argparse
import csv
import keras
import numpy as np
import pickle
from power_consumption_train import load_data
import signal
import sys


def parse_args():
	parser = argparse.ArgumentParser(description="Predict with recurrent network")
	parser.add_argument("-l", "--limit", type=int, help="Maximum number of bytes to load from dataset")
	parser.add_argument("data_path", metavar="data-path", help="The data file to predict from")
	parser.add_argument("model_prefix", metavar="model_prefix", help="Prefix for loading the model")
	return parser.parse_args()


def load_model(path):
	model = keras.models.load_model(path + ".model.keras")
	with open(path + ".model", "rb") as infile:
		normalizer, X_names, y_names = pickle.load(infile)
	return model, normalizer, X_names, y_names


def print_results(X, X_names, y_true, y_names, y_pred):
	y_pred_names = [yn+"_PRED" for yn in y_names]
	header = X_names + y_names + y_pred_names
	data = np.hstack([X, y_true, y_pred])

	writer = csv.writer(sys.stdout, delimiter=str("\t"))
	writer.writerow(header)
	writer.writerows(data)


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	X, y, X_names_data, y_names_data = load_data(args.data_path, args.limit)
	model, normalizer, X_names, y_names = load_model(args.model_prefix)

	if X_names != X_names_data:
		raise ValueError("Input names don't match. Model expects {0}".format(X_names))
	if y_names != y_names_data:
		raise ValueError("Output names don't match. Model produces {0}".format(y_names))

	X = np.reshape(X, (X.shape[0], X.shape[1], 1))
	y_pred = model.predict(X)
	X = np.squeeze(X)

	print_results(X, X_names, y, y_names, y_pred)