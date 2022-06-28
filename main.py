
#TODO

def local_PFA(prefix,logProb):
	# a local PFA: 
	# for each features feat(prefix) * w 
	# for f in 

	return x

def process_data():
	# return 
	return x

def str2fmat(string, attributes):
	# comment: return feature matrix (fmat)
	return fmat

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