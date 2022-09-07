import bilinear

# python ${source}/daland_eval.py ${experiment_dir}/Judgements/${name}_${i} ${source}/data/Daland_etal_2011__AverageScores.csv 

if __name__ == '__main__':
	# training_split = 80
	vectors_filename="data/onset_tokens_arpa_bigram_ppmi_word2vec.w2v"
	train_filename="data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1"
	dev_filename="data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2"
	token_filename="data/onset_tokens_arpa.txt"

	# num_lines = sum(1 for line in open('data/onset_tokens_arpa.txt'))
	
	# split = num_lines // training_split
	
	# with open(train_filename, 'r') as f:
	# 	file_lines = f.readlines()

	# random.shuffle( open('data/onset_tokens_arpa.txt'))
	# dev = file_lines[:len(file_lines) * 0.2]
	# train = file_lines[len(file_lines) * 0.2:]	
	# train_filename
	# dev = 


	# for i in range(1,10):
	# 	bilinear.main(vectors_filename,
	# 		train_filename,
	# 		dev_filename=None,
	# 		test_filename=None,
	# 		no_encoders=True,
	# 		batch_size=	2**i)
		
		
		# --no_encoders ${w2v_path} ${ngrams_path} --batch_size $i --num_iter $((num_sample/i))

	### Fit a MaxEnt model ###

	print("1. Fitting phonotactic grammar")

	bilinear.main(vectors_filename,
		train_filename,
		dev_filename,
		test_filename=None,
		no_encoders=True,
		batch_size=	2)
		


	### Test correlations with Daland Et Al judgements ###
	# echo "2. Testing Daland Et Al correlations"

	# Train the bilinear model on the onset ARPABET token data, trying a range of relevant hyperparameters to find the best fit
# Test the data on the Daland et al onsets and get the output of the model in the format that Max's code expects
# Run Max's code to calculate correlations between our model and the human judgements
# Compare against results from Max's model