# name="English_Template"
# config="../expts/confs/English_SC_onsets.config"
# premade_classes=""
# cluster_on="contexts"
w2v_path="../distributional_learning/vector_data/parupa_bigram_ppmi_word2vec.w2v"
ngrams_path="../distributional_learning/vector_data/parupa_bigram_ppmi_word2vec.ngrams"

source=".." 

### Fit a MaxEnt model ###

echo "1. Fitting phonotactic grammar"


python bilinear.py --no_encoders ${w2v_path} ${ngrams_path} 

### Test correlations with Daland Et Al judgements ###
echo "2. Testing Daland Et Al correlations"

# python ${source}/daland_eval.py ${experiment_dir}/Judgements/${name}_${i} ${source}/data/Daland_etal_2011__AverageScores.csv 

