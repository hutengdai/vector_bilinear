import sys
import csv

import torch

import readwrite as rw
from bilinear import WordVectors, MarginalLogLinear, ConditionalSoftmax, ConditionalLogBilinear

def main(model_filename, data_filename, out_filename):
	
	model = torch.load(model_filename)
	# model.bilinear
	# breakpoint()
	with open(out_filename, "w", newline= "") as outfile:
		writer = csv.writer(outfile)
		with open(data_filename, newline= "") as infile:
			for form in infile:
				word = ['#'] + form.strip().split() + ['#']
				pairs = zip(word, word[1:])
				score = model(pairs).sum()
				writer.writerow([form.strip(), score.item()])



if __name__ == '__main__':
	main(*sys.argv[1:])
	