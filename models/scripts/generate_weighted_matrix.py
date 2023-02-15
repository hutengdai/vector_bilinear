import argparse
import csv
import numpy as np

from nltk import FreqDist
from nltk.lm import KneserNeyInterpolated, NgramCounter
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
from os import path

N = 2

def create_weighted_matrix(dataset, weighting, model_name, outdir, word2vec, discard_duplicates):
    if not model_name:
        # Model name is dataset + weighting if not specified
        model_name = path.split(dataset)[-1].split('.')[0] + "_{}".format(weighting)

    # Read in data and add padding symbols
    with open(dataset, 'r') as f:
        reader = csv.reader(f)
        # Add our own padding symbols
        tokens = [
            ['#'] + token[0].split(' ') + ['#']
            for token in reader
        ]
    if discard_duplicates:
        tokens = list(set(tuple(x) for x in tokens))

    # Count bigrams. These are only used in the w2v output
    bigrams = [gram for token in tokens for gram in ngrams(token, 2)]
    bigram_counts = FreqDist(bigrams)

    # Train forwards LM to get probabilities for X _ contexts
    f_train, f_vocab = padded_everygram_pipeline(N, tokens)
    f_model = KneserNeyInterpolated(N)
    f_model.fit(f_train, f_vocab)

    # Train backwards LM to get probabiliteis for _ X contexts
    reversed_tokens = [list(reversed(token)) for token in tokens]
    b_train, b_vocab = padded_everygram_pipeline(N, reversed_tokens)
    b_model = KneserNeyInterpolated(N)
    b_model.fit(b_train, b_vocab)

    # Create (P)PMI matrix for both models independently
    f_matrix, symbols = create_matrix(f_model, weighting)
    b_matrix, _ = create_matrix(b_model, weighting)

    # Create names of contexts for each model
    f_contexts = ['{}_'.format(s) for s in symbols]
    b_contexts = ['_{}'.format(s) for s in symbols]
    contexts = f_contexts + b_contexts

    # Concatenate forward and backward (P)PMI matrices
    full_matrix = np.concatenate((f_matrix, b_matrix), axis=1)

    # Save matrix to output file
    save_matrix(
        full_matrix, symbols, contexts, model_name, 
        outdir, word2vec, bigram_counts
    )    

def create_matrix(model, weighting):
    # Get list of unique symbols
    symbols = sorted(model.vocab.counts.keys())

    # Removing nltk-provided padding symbols since we don't
    # want embeddings for these
    symbols.remove('<s>')
    symbols.remove('</s>')
    num_syms = len(symbols)

    # Initialize probability matrix
    matrix = np.zeros([num_syms, num_syms])

    # Each cell in this matrix is P(s|c)
    # If the <s> and </s> symbols were included the columns of the matrix
    # would each sum to 1,  
    for i, s in enumerate(symbols):
        for j, c in enumerate(symbols):
            matrix[i,j] = model.score(s, [c])

    # Renormalize by column since we removed the boundary symbols
    matrix = matrix / matrix.sum(axis=0)

    # Covnert matrix to (P)PMI
    w_matrix = weight_matrix(matrix, weighting)

    return w_matrix, symbols

def weight_matrix(matrix, weighting):
    # Input matrix is P(s|c)
    # Get P(s) (= P(c)) by summing over all contexts
    p_s = matrix.sum(axis=1).reshape(-1, 1) / matrix.sum()

    # Then convert to P(s,c) = P(s|c) * P(c)
    matrix *= p_s.transpose()

    # Then compute P(s) * P(c)
    denominator = p_s * p_s.transpose()

    # Calculate PMI
    pmi_matrix = np.log(matrix / denominator)

    # Convert to PPMI if needed
    if weighting == 'pmi':
        result = pmi_matrix
    elif weighting == 'ppmi':
        result = np.maximum(pmi_matrix, 0)
    else:
        raise "Unknown weighting {}".format(weighting)

    return result

def save_matrix(matrix, symbols, contexts, model_name, outdir, word2vec, 
                bigram_counts):
    if not word2vec:
        # Don't produce embedding of word boundary unless we're
        # using word2vec output
        matrix = np.delete(matrix, symbols.index('#'), axis=0)
        symbols.remove('#')

        np.savetxt(path.join(
            outdir, '{}.data'.format(model_name)), matrix, fmt='%f'
        )
        with open(path.join(outdir, '{}.sounds'.format(model_name)), 'w') as f:
            print(' '.join(symbols), file=f)
        with open(path.join(outdir, '{}.contexts'.format(model_name)), 'w') as f:
            print(' '.join(contexts), file=f)
    else:
        # Write embedding file in word2vec format
        with open(path.join(outdir, '{}.w2v'.format(model_name)), 'w') as f:
            f.write("{} {}\n".format(*matrix.shape))
            for i, sound in enumerate(symbols):
                embedding = matrix[i]
                row = '{} {}\n'.format(sound, ' '.join(map(str, embedding)))
                f.write(row)

        # Write ngram counts file
        with open(path.join(outdir, '{}.ngrams'.format(model_name)), 'w') as f:
            for ngram, count in bigram_counts.items():
                row = ','.join(list(ngram) + [str(count)]) + '\n'
                f.write(row)

        # Record contexts too in case we need them later
        with open(path.join(outdir, '{}.contexts'.format(model_name)), 'w') as f:
            print(' '.join(contexts), file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a vector space embedding of segments in a "
                    "phonological data set."
    )
    parser.add_argument(
        'dataset', type=str, help='The corpus to vectorize.'
    )
    parser.add_argument(
        '--weighting', default='pmi', type=str,
        help='The method to weight the raw counts'
    )
    parser.add_argument(
        '--outfile', type=str, default=None,
        help='The filename to save the vector model under.'
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help='The directory to save the vector data in.'
    )
    parser.add_argument(
        '--word2vec', action="store_true",
        help='Output embeddings will be in word2vec format'
    )
    parser.add_argument(
        '--discard_duplicates', action="store_true",
        help="Throw out duplicate tokens"
    )

    args = parser.parse_args()
    create_weighted_matrix(
        args.dataset, args.weighting, args.outfile, args.outdir, 
        args.word2vec, args.discard_duplicates
    )