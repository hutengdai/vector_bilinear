import csv

# Single character maps
IPA_ARPA_MAP = {
    'b': 'B',
    'd': 'D',
    'f': 'F',
    'g': 'G',
    'k': 'K',
    'l': 'L',
    'm': 'M',
    'n': 'N',
    'p': 'P',
    'ɹ': 'R',
    's': 'S',
    't': 'T',
    'v': 'V',
    'w': 'W',
    'j': 'Y',
    'z': 'Z',
    'ɑ': 'AA',
    'æ': 'AE',
    'ʌ': 'AH',
    'ɔ': 'AO',
    'ɛ': 'EH',
    'ɝ': 'ER',
    'ɪ': 'IH',
    'i': 'IY',
    'o': 'OW',
    'ʊ': 'UH',
    'u': 'UW',
    'ð': 'DH',
    'h': 'HH',
    'ŋ': 'NG',
    'ʃ': 'SH',
    'θ': 'TH',
    'ʒ': 'ZH',
    'e': 'EY',
    'aʊ': 'AW',
    'aɪ': 'AY',
    'eɪ': 'EY',
    'ɔɪ': 'OY',
    'tʃ': 'CH',
    'dʒ': 'JH'
}

def convert_features_trinary(feature):
    if feature == '+':
        return 1
    elif feature == '−':
        return -1
    elif feature == '0':
        return 0
    else:
        return IPA_ARPA_MAP[feature]

def convert_features_binary(feature):
    # Output is one cell for each feature/value combo,
    # so 2 cols per feature. 
    if feature == '+':
        return [1, 0]
    elif feature == '−':
        return [0, 1]
    elif feature == '0':
        return [0, 0]
    else:
        return IPA_ARPA_MAP[feature]

def create_binary_contexts(feature_name):
    if feature_name:
        return [feature_name + '_plus', feature_name + '_minus']
    else:
        return ['']

with open('data/features/english_features.csv', 'r') as f:
    reader = csv.reader(f)

    trinary_contexts = next(reader)
    binary_contexts = [x for feat in map(create_binary_contexts, trinary_contexts) for x in feat]
    binary_output = []
    trinary_output = []

    for line in reader:
        binary_row = [x for feat_set in map(convert_features_binary, line) for x in feat_set]
        trinary_row = list(map(convert_features_trinary, line))
        binary_output.append(binary_row)
        trinary_output.append(trinary_row)
    
with open('data/features/english_trinary_features_columns.txt', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(trinary_contexts)

with open('data/features/english_binary_features_columns.txt', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(binary_contexts)

with open('data/features/english_binary_features.w2v', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(binary_output)

with open('data/features/english_trinary_features.w2v', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(trinary_output)
