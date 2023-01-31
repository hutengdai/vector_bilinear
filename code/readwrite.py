import sys
import csv
import random
import subprocess
from collections import Counter

import tqdm

INF = float('inf')

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

def read_numbers(filename):
    lc = line_count(filename)
    def gen():
        with open(filename) as infile:
            print("Reading numbers from %s ..." % filename, file=sys.stderr)
            reader = csv.reader(infile)
            for *parts, count in tqdm.tqdm(reader, total=lc):
                count = float(count)
                if count == INF or count == -INF:
                    print("Bad line: %s" % str(parts + [count]), file=sys.stderr)
                yield tuple(parts), count
    return dict(gen())

def read_counts(filename, verbose=False):
    lc = line_count(filename)
    def gen():
        with open(filename) as infile:
            print("Reading counts from %s ..." % filename, file=sys.stderr)
            reader = csv.reader(infile)
            for *parts, count in tqdm.tqdm(reader, total=lc):
                count = int(count)
                yield tuple(parts), count
    result = Counter(dict(gen()))
    N = sum(result.values())
    if verbose:
        print("Total %d" % N, file=sys.stderr)
    return result

def read_groups(filename):
    lc = line_count(filename)
    with open(filename) as infile:
        reader = csv.reader(infile)
        yield from tqdm.tqdm(reader)
    
def write_counts(outfile, items):
    writer = csv.writer(outfile)
    for parts, count in items:
        row = list(parts) + [count]
        writer.writerow(row)

def read_vectors(filename):
    print("Loading vectors from %s ..." % filename, file=sys.stderr)
    d = {}
    with open(filename) as infile:
        n, ndim = next(infile).strip().split()
        n = int(n)
        ndim = int(ndim)
        lines = list(infile)
        for line in tqdm.tqdm(lines):
            parts = line.strip().split(" ")
            numbers = list(map(float, parts[-ndim:]))
            wordparts = parts[:-ndim]
            word = " ".join(wordparts)
            d[word] = numbers
    return d

def read_words(filename):
    print("Reading words from %s ..." % filename, file=sys.stderr)
    lc = line_count(filename)
    with open(filename) as infile:
        for line in tqdm.tqdm(infile, total=lc):
            yield line.strip()
