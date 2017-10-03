#!/usr/bin/env python3

import argparse
import csv
import fileinput
import numpy as np
import signal
import sklearn.metrics
import sys

def parse_args():
	parser = argparse.ArgumentParser(description="Predict with recurrent network")
	parser.add_argument("-m", "--mode", required=True, choices=["auc", "mae"], help="Which score to compute")
	parser.add_argument("path", metavar="path", help="The data file to score")
	return parser.parse_args()


def load_data(path):
	infile = fileinput.FileInput(path, openhook=fileinput.hook_compressed)
	infile = map(bytes.decode, infile) # fileinput seems to ignore mode and always read binary?
	reader = csv.DictReader(infile, delimiter=str("\t"))
	header = reader.fieldnames

	y_names, y_indices = zip(*[(t, idx) for idx, t in enumerate(header) if t.endswith("_TARGET")])
	y_pred_names, y_pred_indices = zip(*[(p, idx) for idx, p in enumerate(header) if p.endswith("_TARGET_PRED")])

	y = np.loadtxt(path, delimiter=str("\t"), skiprows=1, usecols=y_indices)
	y_pred = np.loadtxt(path, delimiter=str("\t"), skiprows=1, usecols=y_pred_indices)

	return y, y_names, y_pred, y_pred_names


if __name__ == "__main__":
	signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	args = parse_args()

	y, y_names, y_pred, y_pred_names = load_data(args.path)

	if args.mode == "auc":
		raise NotImplementedError("AUC Score not implemented!")
	elif args.mode == "mae":
		scores = sklearn.metrics.mean_absolute_error(y, y_pred, multioutput="raw_values")

	writer = csv.writer(sys.stdout, delimiter=str("\t"), lineterminator=str("\n"))
	writer.writerow("target score".split())
	for yn, s in zip(y_names, scores):
		writer.writerow([yn, s])