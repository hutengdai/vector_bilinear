import csv

def convert_features_trinary(feature):
    if feature == '+':
        return 1
    elif feature == '−' or feature == '-':
        return -1
    elif feature == '0':
        return 0
    else:
        return feature

def convert_features_binary(feature):
    # Output is one cell for each feature/value combo,
    # so 2 cols per feature. 
    if feature == '+':
        return [1, 0]
    elif feature == '−' or feature == '-':
        return [0, 1]
    elif feature == '0':
        return [0, 0]
    else:
        return [feature]

def create_binary_contexts(feature_name):
    if feature_name:
        return [feature_name + '_plus', feature_name + '_minus']
    else:
        return ['']

with open('models/embeddings/discrete_distributional_features.csv', 'r') as f:
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
    
# with open('models/embeddings/learned_trinary_features_columns.txt', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(trinary_contexts)

# with open('models/embeddings/learned_trinary_features.w2v', 'w') as f:
#     writer = csv.writer(f, delimiter=' ')
#     writer.writerow([len(trinary_output), len(trinary_output[0]) - 1])
#     writer.writerows(trinary_output)


with open('models/embeddings/discrete_distributional_features_columns.txt', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(binary_contexts)

with open('models/embeddings/discrete_distributional_features.w2v', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerow([len(binary_output), len(binary_output[0]) - 1])
    writer.writerows(binary_output)