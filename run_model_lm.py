import sys
import csv

import torch

import readwrite as rw
from bilinear import WordVectors, MarginalLogLinear, ConditionalSoftmax, ConditionalLogBilinear

def main(model_filename, data_filename):
	model = torch.load(model_filename)
	writer = csv.writer(sys.stdout)    
	with open(data_filename) as infile:
		for form in infile:
			word = ['#'] + form.strip().split() + ['#']
			pairs = zip(word, word[1:])
			score = model(pairs).sum()
			writer.writerow([form.strip(), score.item()])

if __name__ == '__main__':
	main(*sys.argv[1:])
	