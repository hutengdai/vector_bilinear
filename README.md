# DFM

This is the repo for the collaborative project of Huteng Dai, Richard Futrell and Connor Mayer on modelling
phonotactics from large-scale corpus. We are currently testing our hypothesis on judgment data of English nonce words (Daland et al. 2011).

The goal of the study is to examine the role of natural classes in phonological learning. We ask: is it possible for a learner to acquire phonotactic grammar without prespecified featural system that is assumed in previous proposals. Moreover, is it possible to simultaneously induce a system of natural class and a phonotactic grammar?

Here is how to run the program:

- for binary features
python3 run_learning_model.py data/english_binary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2

- for ternary features
python3 run_learning_model.py data/english_ternary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2

- for learned embeddings (Mayer 2020)
python run_learning_model.py data/onset_tokens_arpa_bigram_ppmi_word2vec.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2

(Remote machine users: if git is not syncing, quit remote machine, and try:
scp huteng@sephiroth.socsci.uci.edu:~/filename . 
)

rerun command:

<!-- binary feature -->
python bilinear.py data/english_binary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/binary_feature_10_27.pt

<!-- ternary feature -->
python bilinear.py data/english_trinary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/ternary_feature_10_27.pt

python bilinear.py data/onset_tokens_arpa_bigram_ppmi_word2vec.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.01 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/induced_ppmi_class_10_27.pt

python bilinear.py data/onset_tokens_arpa_bigram_pmi_word2vec.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/induced_pmi_class_10_27.pt


<!-- Run saved models -->

python run_saved_model.py result/binary_feature_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/binary_feature_10_27.txt

python run_saved_model.py result/ternary_feature_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/ternary_feature_10_27.txt

python run_saved_model.py result/induced_ppmi_class_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/induced_ppmi_class_10_27.txt

python run_saved_model.py result/induced_pmi_class_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/induced_pmi_class_10_27.txt


Analysizing testing result:

load pytorch model



headmap for constraints A matrix

black and white for feature matrices.





