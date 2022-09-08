from platform import machine
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statannot import add_stat_annotation
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from plotnine import ggplot, geom_violin, aes, stat_smooth, facet_wrap
import pprint
import numpy as np

pp = pprint.PrettyPrinter(indent=4)
pprint.sorted = lambda x, key=None: x


def visual_judgefile(humanjudgefile,bilinear_judgement):

	data = pd.read_csv(
		humanjudgefile,
		sep=",",
		header=0,
		encoding="utf-8")
	machine_data = pd.read_csv(
		bilinear_judgement,
		sep=",",
		# header=0,
		names=["onset","score"],
		encoding="utf-8")

	# fig_human = (ggplot(data, aes('attestedness', 'likert_rating')) 
	# + geom_violin(data))
	# dir(fig)
	# fig_human.save('daland.png', dpi=300)

	data["machine_judgement"] = machine_data["score"]

	print(data)
	
	pearsoncorr, _ = pearsonr(data['likert_rating'], data['machine_judgement'])
	print('Pearsons correlation: %.3f' % pearsoncorr)
	spearmancorr, _ = spearmanr(data['likert_rating'], data['machine_judgement'])
	print('Spearman correlation: %.3f' % spearmancorr)
	fig = sns.scatterplot(data=data, x="machine_judgement", y="likert_rating")
	plt.savefig('correlation.png', dpi=300)

	# dir(fig)
	# + stat_smooth(method='lm')
	# + facet_wrap('~gear')
	# plt.show()
	
if __name__ == '__main__':
	# JudgementFile = 'data\\TurkishJudgement-MaxEnt-MaxOE1.txt'
	# JudgementFile = 'data\\TurkishJudgement-categorical.txt'
	# JudgementFile = 'data\\TurkishJudgement-probabilistic-100ep005lr.txt'
	# JudgementFile = 'data\\TurkishJudgementFile-tolerance.txt'
	# visual_judgefile_maxent(JudgementFile)


	humanJudgement = "data\\Daland_etal_2011__AverageScores.csv"
	bilinear_judgement = "data\\bilinear_judgement.txt"
	nelson_judgement = "data\\Nelson_model_onset_judgement.txt"
	visual_judgefile(humanJudgement,nelson_judgement)
