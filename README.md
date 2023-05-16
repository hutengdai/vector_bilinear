# DFM Project

This repository hosts a collaborative project by Huteng Dai, Richard Futrell and Connor Mayer, focusing on phonotactics modelling using large-scale corpus data. The project revolves around testing our hypothesis on judgment data of English nonce words based on the study by Daland et al. in 2011.

## Project Aim

The purpose of this study is to investigate the role of natural classes in phonological learning. Specifically, we ask whether a learner can acquire phonotactic grammar without a prespecified featural system as assumed in previous proposals. Furthermore, we are interested in whether it's possible to induce a system of natural class and a phonotactic grammar simultaneously.

## How to Run the Program

For users of remote machines, if git is not syncing, quit the remote machine and try the following command:
```
scp huteng@sephiroth.socsci.uci.edu:~/filename .
```

### Running the Learning Model

The learning model can be run using different feature sets. Below are examples of how to run the model with binary, ternary, and various other features.

Run command format:
```
python bilinear.py [input feature file] [input ngram file] --dev [dev file] --lr [learning rate] --batch_size [batch size] --no_encoders --num_iter [number of iterations] --output_filename [output file]
```

Example for binary feature:
```
python bilinear.py ./input/english_binary_features.w2v ./input/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev ./input/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename ./result/binary_feature_10_27.pt
```

Example for ternary feature:
```
python bilinear.py ./input/english_ternary_features.w2v ./input/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_1 --dev ./input/onset_tokens_arpa_bigram_ppmi_word2vec.ngrams_2 --lr 0.001 --batch_size 64 --no_encoders --num_iter 881 --output_filename ./result/ternary_feature_10_27.pt
```

### Running Saved Models

The saved models can be run using the following command format:
```
python run_saved_model.py [saved model file] [input test data file] [output file]
```

Example:
```
python run_saved_model.py ./result/binary_feature_10_27.pt ./input/test_data_daland_et_al_arpa_onset_only.txt ./result/binary_feature_10_27.txt
```

### Analyzing Testing Results

For analyzing the testing results, we load the PyTorch model and create heatmaps for the constraints A matrix and feature matrices. The heatmaps are represented in black and white.

#### Note

- The X-axis in the heatmap represents the preceding context.
- The binary and ternary feature files (w2v) miss features for the word boundary.


