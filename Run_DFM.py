import bilinear
import pandas as pd
import run_model_lm
# python ${source}/daland_eval.py ${experiment_dir}/Judgements/${name}_${i} ${source}/data/Daland_etal_2011__AverageScores.csv 

if __name__ == '__main__':
	# training_split = 80
	# vectors_filename="data/onset_tokens_arpa_bigram_ppmi_word2vec.w2v"
	vectors_filename="data/features/english_binary_features.w2v"
	train_filename="data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1"
	dev_filename="data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2"
	# token_filename="data/onset_tokens_arpa.txt"
	
	import argparse

	parser = argparse.ArgumentParser(description='Estimate conditional word probabilities using a log-bilinear model')
	parser.add_argument("feature_embedding", type=str, help="feature embedding file path")
	parser.add_argument("training_file", type=str, help="Training file")
	parser.add_argument("dev_file", type=str, help="Dev file")
	# parser.add_argument("testing_file", type=str, help="Test file")

	# parser.add_argument("", type=str, help="")

	args = parser.parse_args()
	vectors_filename = args.feature_embedding
	train_filename = args.training_file
	dev_filename = args.dev_file


	training_data = pd.read_csv(train_filename, header = None)
	training_data_size = sum(training_data[2])
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
	output_filename= bilinear.DEFAULT_FILENAME

	# model, diagnostics = bilinear.main(vectors_filename,
	# 				train_filename,
	# 				dev_filename,
	# 				test_filename=None,
	# 				no_encoders=True,
	# 				batch_size=	32,
	# 				lr = 0.1,
	# 				# num_iter=200,
	# 				num_iter = 200,
	# 				output_filename=output_filename
	# 				)
	# 			# breakpoint()
	# with open("hyperparameters\\diagnostics_print%s_%s.txt" % (str(32),str(0.1)), "w") as diagnostics_print:
	# 	# for row in diagnostics:
	# 	a = pd.DataFrame(diagnostics)
	# 	print(a)
	# 	a["batch_size"] = 32
	# 	a["lr"] = 0.1
	# 	a["num_iter"] = 200

	# 	breakpoint()

	df = pd.DataFrame()
	for batch_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
		for lr in [0.1, 0.01, 0.001, 0.0001]:
			model, diagnostics = bilinear.main(vectors_filename,
				train_filename,
				dev_filename,
				test_filename=None,
				no_encoders=True,
				batch_size=	batch_size,
				lr = lr,
				# num_iter=200,
				num_iter = int(training_data_size / batch_size),
				output_filename=output_filename
				)
			# breakpoint()
			with open("hyperparameters\\diagnostics_print%s_%s.txt" % (str(batch_size),str(lr)), "w") as diagnostics_print:
				# for row in diagnostics:
				a = pd.DataFrame(diagnostics)
				print(a)
				a["batch_size"] = batch_size
				a["lr"] = lr
				a["num_iter"] = int(training_data_size / batch_size)
				df = df.append(a)
				# diagnostics_print.write(row+'\n')
				# print(row, file=diagnostics_print)

# control R
	# dalandfile = "data\\Daland_et_al_arpa_onset_only.txt"
	# out_filename = "data\\bilinear_judgement.txt"

	## Test correlations with Daland Et Al judgements ###
	# run_model_lm.main(output_filename,dalandfile,out_filename)


	# echo "2. Testing Daland Et Al correlations"

# Train the bilinear model on the onset ARPABET token data, trying a range of relevant hyperparameters to find the best fit
# Test the data on the Daland et al onsets and get the output of the model in the format that Max's code expects
# Run Max's code to calculate correlations between our model and the human judgements
# Compare against results from Max's model


