import sys
import csv
import random
import itertools
import functools
import argparse

import numpy as np
import scipy.special
import pandas as pd
import torch

try:
    import genbmm
except ModuleNotFoundError:
    genbmm = None

try:
    import entmax
except ModuleNotFoundError:    
    entmax = None

LOG2 = np.log(2)
BOUNDARY_SYMBOL_INDEX = 0
INITIAL_STATE_INDEX = 0
INF = float('inf')
EOS = '!!</S>!!'

EPSILON = 10 ** -8
DEFAULT_NUM_EPOCHS = 10 ** 5
DEFAULT_NUM_STATES = 500
DEFAULT_BATCH_SIZE = 5
DEFAULT_PRINT_EVERY = 1000
DEFAULT_NUM_SAMPLES = 0
DEFAULT_DATA_SEED = 0
DEFAULT_INIT_TEMPERATURE = 1
DEFAULT_PERM_TEST_NUM_SAMPLES = 0
DEFAULT_ACTIVATION = "softmax" # sparsemax and entmax15 are available but numerically unstable
DEFAULT_LR = 0.001

#TODO

def local_PFA(prefix,logProb):
	# a local PFA: 
	# for each features feat(prefix) * w 
	# for f in 

	return x


def str2fmat(string, attributes):
	# comment: return feature matrix (fmat)
	return fmat


def get_txt_corpus_data(filename):
    """
    Reads input file and coverts it to list of lists. Word boundaries will be added later.
    """
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            yield line.strip().split(' ')

def shuffled(xs):
    xs = list(xs)
    random.shuffle(xs)
    return xs

def process_data():
	# return 
	return x

def main(input_file,
         test_file,
         model_class=DEFAULT_NUM_STATES,
         num_epochs=DEFAULT_NUM_EPOCHS,
         num_samples=DEFAULT_NUM_SAMPLES,
         print_every=DEFAULT_PRINT_EVERY,
         seed=DEFAULT_DATA_SEED,
         perm_test_num_samples=DEFAULT_PERM_TEST_NUM_SAMPLES,
         **kwds):
    # model_class is either 'sp', 'sl', 'sp_sl', or a number of states
    first_line = True
    random.seed(seed)
    data = shuffled(get_txt_corpus_data(input_file))
    phone2ix, ix2phone, training_data, dev = process_data(data)
    num_symbols = len(ix2phone)
    d = kwds.copy()
    print("Training data = ", input_file, file=sys.stderr)
    print("Nonce testing data = ", test_file, file=sys.stderr)
    print("Training set size =", len(training_data), file=sys.stderr)
    print("Dev set size =", len(dev), file=sys.stderr)
    print("Segment inventory size = ", len(ix2phone), file=sys.stderr)


if __name__ == '__main__':
	FeatureFile = 'data\\TurkishFeatures-tell.txt'
	TrainingFile = 'data\\TurkishLearningData-tell.txt'
	TestingFile = 'data\\TurkishTestingData.txt'
	sample = get_corpus_data(TrainingFile)
	alphabet, max_length = process_data(sample)
	ix2phone = {ix: p for (ix, p) in enumerate(alphabet)}
	phone2ix = {p: ix for (ix, p) in enumerate(alphabet)}
	feat, feature_dict, num_feats, feature_table, feat2ix, ix2feat = process_features(FeatureFile, alphabet, ix2phone)
	vowel = [x for x in feature_dict if feature_dict[x][feat2ix['syll']] == "+" if feature_dict[x][feat2ix['long']] == "-"] #