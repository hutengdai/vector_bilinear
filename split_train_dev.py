import sys
import random
from collections import Counter

import readwrite as rw

def resample_counts(histogram, k):
    elements = list(histogram.elements())
    random.shuffle(elements)
    one, two = elements[:-k], elements[-k:]
    return Counter(one), Counter(two)

def main(filename, dev_proportion, seed=None):
    if seed is not None:
        random.seed(seed)
    c = rw.read_counts(filename)
    N = sum(c.values())
    print("N = %d" % N, file=sys.stderr)
    proportion = 1-float(dev_proportion)
    num_train = int(proportion*N)
    num_dev = N - num_train
    train, dev = resample_counts(c, num_dev)

    print(train)

    fn1 = filename+"_1"
    print("Writing N_1 = %d datapoints to %s" % (num_train, fn1), file=sys.stderr)
    with open(fn1, 'wt', newline='') as outfile:
        rw.write_counts(outfile, train.items())

    fn2 = filename+"_2"
    print("Writing N_2 = %d datapoints to %s" % (num_dev, fn2), file=sys.stderr)        
    with open(fn2, 'wt', newline='') as outfile:
        rw.write_counts(outfile, dev.items())

if __name__ == '__main__':
    # this code will generate training and dev set
    main(*sys.argv[1:])
