import bilinear
import pandas as pd
import run_saved_model
import datetime


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
	parser.add_argument("result_file", type=str, help="Result file")

	# parser.add_argument("testing_file", type=str, help="Test file")

	# parser.add_argument("", type=str, help="")

	args = parser.parse_args()
	vectors_filename = args.feature_embedding
	train_filename = args.training_file
	dev_filename = args.dev_file
	result_file = args.result_file


	training_data = pd.read_csv(train_filename, header = None)
	training_data_size = sum(training_data[2])

	# num_lines = sum(1 for line in open('data/onset_tokens_arpa.txt'))


	print("1. Fitting phonotactic grammar")
	output_filename= bilinear.DEFAULT_FILENAME
	header = True
	for batch_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
		
		for lr in [0.1, 0.01, 0.001, 0.0001]:
			model, diagnostics = bilinear.main(vectors_filename,
				train_filename,
				dev_filename,
				test_filename=None,
				no_encoders=True,
				batch_size=	batch_size,
				lr = lr,
				check_every = 1,
				num_iter = int(training_data_size / batch_size),
				output_filename=output_filename
				)
			a = pd.DataFrame(diagnostics)
			a["batch_size"] = batch_size
			a["lr"] = lr
			a["num_iter"] = int(training_data_size / batch_size)
			a.to_csv("result/%s.csv" % str(result_file), mode='a+', index=False, header=header)
			header = False
			print("Loop is finished! Batch size %s Learning rate%s" %(str(batch_size), str(lr)))
