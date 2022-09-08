import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statannot import add_stat_annotation
import numpy as np
import pandas as pd
from scipy.special import softmax
from plotnine import ggplot, geom_violin, aes, stat_smooth, facet_wrap
import pprint
import numpy as np

pp = pprint.PrettyPrinter(indent=4)
pprint.sorted = lambda x, key=None: x


def visual_judgefile(judgefile):

	data = pd.read_csv(
		judgefile,
		sep=",",
		header=0,
		# names=["word", "hx", "log likelihood"],

		encoding="utf-8")
	print(data)
	fig = (ggplot(data, aes('attestedness', 'score')) 
	+ geom_violin(data))

	# fig.show()
	fig.savefig('daland.png', dpi=300)


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
	visual_judgefile(humanJudgement)
