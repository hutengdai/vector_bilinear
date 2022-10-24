# DFM

This is the repo for the collaborative project of Huteng Dai, Richard Futrell and Connor Mayer on modelling
phonotactics from large-scale corpus. We are currently testing our hypothesis on judgment data of English nonce words (Daland et al. 2011).

The goal of the study is to examine the role of natural classes in phonological learning. We ask: is it possible for a learner to acquire phonotactic grammar without prespecified featural system that is assumed in previous proposals. Moreover, is it possible to simultaneously induce a system of natural class and a phonotactic grammar?

Here is how to run the program:

- for binary features
python3 Run_DFM.py data/features/english_binary_features.w2v data/onset_tokens_arpa_bigram_ppmi_word
2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2

- for learned embeddings (Mayer 2020)
python3 Run_DFM.py data/onset_tokens_arpa_bigram_ppmi_word2vec.w2v data/onset_tokens_arpa_bigram_ppmi_word
2vec.ngrams_1 data/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2

