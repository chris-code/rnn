#!/usr/bin/env python3

import argparse
import csv
import keras
import numpy as np
from pg_train import load_data
import signal
import sys


def parse_args():
	parser = argparse.ArgumentParser(description="Train recurrent network")
	parser.add_argument("-l", "--limit", type=int, help="Maximum number of data points to load from dataset")
	parser.add_argument("-t", "--num-targets", type=int, dest="num_targets", default=1, help="Length of the target sequence")
	parser.add_argument("-o", "--output_length", type=int, dest="output_length", default=30, help="How many sequence items to predict")
	parser.add_argument("data", help="Test data")
	parser.add_argument("model", help="Where the model is located")
	return parser.parse_args()


def load_model(path):
	return keras.models.load_model(path)


def predict(model, X, num_steps):
	preds = np.zeros((X.shape[0], num_steps, X.shape[2]))

	for idx_seq, sequence in enumerate(X):
		sequence = np.expand_dims(sequence, axis=0)

		for step_idx in range(num_steps):
			pred = model.predict(sequence)
			preds[idx_seq, step_idx] = pred[0]

			pred = np.expand_dims(pred, axis=1)
			sequence = np.concatenate((sequence, pred), axis=1)[:,1:,:] # Append new prediction and discard first element

	return preds


def devectorize(preds, idx_to_char):
	preds = np.argmax(preds, axis=2)

	preds_text = []
	for pred in preds:
		pt = ""
		for char_idx in pred:
			pt += idx_to_char[char_idx]
		preds_text.append(pt)

	return preds_text


def print_predictions(X_text, preds_text):
	writer = csv.writer(sys.stdout, delimiter=str("\t"), lineterminator=str("\n"))
	writer.writerow("input output".split())
	writer.writerows(zip(X_text, preds_text))


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	model = load_model(args.model)
	X, y, idx_to_char = load_data(args.data, args.num_targets, limit=args.limit)

	preds = predict(model, X, args.output_length)
	preds_text = devectorize(preds, idx_to_char)
	X_text = devectorize(X, idx_to_char)

	print_predictions(X_text, preds_text)