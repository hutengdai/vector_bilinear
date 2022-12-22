# DFM

This is the repo for the collaborative project of Huteng Dai, Richard Futrell and Connor Mayer on modelling
phonotactics from large-scale corpus. We are currently testing our hypothesis on judgment data of English nonce words (Daland et al. 2011).

The goal of the study is to examine the role of natural classes in phonological learning. We ask: is it possible for a learner to acquire phonotactic grammar without prespecified featural system that is assumed in previous proposals. Moreover, is it possible to simultaneously induce a system of natural class and a phonotactic grammar?

Here is how to run the program:

(Remote machine users: if git is not syncing, quit remote machine, and try:
scp huteng@sephiroth.socsci.uci.edu:~/filename . )

(1) run the learning model

<!-- binary feature -->
python bilinear.py data/english_binary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/binary_feature_10_27.pt

<!-- ternary feature -->
python bilinear.py data/english_ternary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/ternary_feature_10_27.pt

python bilinear.py data/onset_tokens_arpa_bigram_ppmi_word2vec.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.01 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/induced_ppmi_class_10_27.pt

python bilinear.py data/onset_tokens_arpa_bigram_pmi_word2vec.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename result/induced_pmi_class_10_27.pt

<!--  OLD command (output everything in the terminal to binary1.csv):
- for binary features
python run_learning_model.py data/english_binary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 result/binary1.csv

- for ternary features
python run_learning_model.py data/english_ternary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 result/ternary1.csv

- for learned embeddings (Mayer 2020)
python run_learning_model.py data/onset_tokens_arpa_bigram_ppmi_word2vec.w2v data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 result/induced_ppmi_class.csv -->

<!-- Run saved models -->

python run_saved_model.py result/binary_feature_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/binary_feature_10_27.txt
python run_saved_model.py result/ternary_feature_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/ternary_feature_10_27.txt
python run_saved_model.py result/induced_ppmi_class_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/induced_ppmi_class_10_27.txt
python run_saved_model.py result/induced_pmi_class_10_27.pt data/test_data_daland_et_al_arpa_onset_only.txt result/induced_pmi_class_10_27.txt


    Analysizing testing result:

    load pytorch model

    headmap for constraints A matrix

    black and white for feature matrices.


    Missing features for word boundary in binary and ternary feature files (w2v)

    Note: 
    - X-axis in the heatmap indicates the preceding context.

    - New learning data (with exotic words); Polish data (where can we find it)
    - Trigram <- bigram linear model? Tensor sum?


Ternary -> Tied
Binary -> untied

python heatmap.py result/ternary_feature_10_27.pt 1



TODO:
1. Compare tied and untied 
2. attestedness
3. Get correlations for all the models we're testing and put them in a table
4. Break down model scores across three attestedness categories
