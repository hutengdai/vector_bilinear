import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statannot import add_stat_annotation
import numpy as np
import pandas as pd
from scipy.special import softmax
import seaborn as sns
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
pprint.sorted = lambda x, key=None: x


def visual_judgefile(judgefile):

	data = pd.read_csv(
		judgefile,
		sep=",",
		# header=0,
		# names=["word", "hx", "log likelihood"],
		names=[
			"word", "Grammaticality",  "Score","likert_rating","score","phono_cmu","SON"
		],
		encoding="utf-8")
	pp.pprint(data)	
	# pd.to_numeric(data["probability"], downcast="float")
	data.loc[data['Grammaticality'] == 'attested', 'Judgement'] = 1.0
	data.loc[data['Grammaticality'] == 'unattested', 'Judgement'] = 0.0
	pp.pprint(data)
	pearsoncorr, _ = pearsonr(data['Score'], data['Judgement'])
	print('Pearsons correlation: %.3f' % pearsoncorr)
	spearmancorr, _ = spearmanr(data['Score'], data['Judgement'])
	print('Spearman correlation: %.3f' % spearmancorr)
	sns.scatterplot(data=data, x="Score", y="Judgement")

	sns.set_palette("deep", 7)
	sns.set_style('ticks', {"xtick.major.size": 8, "ytick.major.size": 8})
	sns.set(font='Libertinus')

	ax = sns.violinplot(x='Grammaticality', y='Score', data = data, cut = 0)
	ax.set(xlim=(-0.5, 1.5), ylim=(0, 1.25))
	ax.set(xlabel=None)

	
	# ax.set(xlim=(-0.6, 2.6), ylim=(10, 39))
	# ax.set(xlim=(-0.6, 2.6), ylim=(14, 26))
	# ax.set_title('scale = {}'.format(""), y=1)

	add_stat_annotation(
		ax,
		data=data,
		x="Grammaticality",
		y="Score",
		width=0.2,
		# box_pairs=[("ejective-ejective", "legal"),
		#            ("plain-ejective", "legal")],
		# box_pairs=[("stop-aspirate", "legal"),
		#            ("stop-ejective", "legal"),
		#            ("stop-ejective", "stop-aspirate")],
		box_pairs=[("grammatical", "ungrammatical"),
				#    ("illegal-ejective", "illegal-aspirate")
					],
		# test='Kruskal',
		test='Mann-Whitney-ls',
		# test='Levene',
		text_format='full',
		loc='inside',
		verbose=1)

	plt.savefig('Judgement.png', bbox_inches="tight", dpi=600)  # Option 3

	# plt.show()
	

def visual_judgefile_maxent(judgefile):

	data = pd.read_csv(
		judgefile,
		sep="\t",
		header=0,
		# names=[
		# 	"Word", "acceptability", "perplexity", "probability"
		# ],
		encoding="utf-8")

	Z = sum(np.exp(-data['Score']))
	data['Probability'] = np.exp(-data['Score'])/Z
	data.loc[data['Grammaticality'] == 'grammatical', 'Judgement'] = 1.0
	data.loc[data['Grammaticality'] == 'ungrammatical', 'Judgement'] = 0.0
	pp.pprint(data)
	pearsoncorr, _ = pearsonr(data['Probability'], data['Judgement'])
	print('Pearsons correlation: %.3f' % pearsoncorr)
	spearmancorr, _ = spearmanr(data['Probability'], data['Judgement'])
	print('Spearman correlation: %.3f' % spearmancorr)
	sns.scatterplot(data=data, x="Probability", y="Judgement")
	# plt.show()

	# pd.to_numeric(data["probability"], downcast="float")
	sns.set_palette("deep", 7)
	sns.set_style('ticks', {"xtick.major.size": 8, "ytick.major.size": 8})
	sns.set(font='Libertinus')
	ax = sns.violinplot(x='Grammaticality', y='Probability', data = data, cut = 0)
	ax.set(xlabel=None)
	# ax.set(xlim=(-0.5, 1.5), ylim=(-0.25, 1.25))
	ax.set(xlim=(-0.5, 1.5), ylim=(0.0000, 0.0017))
	# ax.set(xlim=(-0.6, 2.6), ylim=(10, 39))
	# ax.set(xlim=(-0.6, 2.6), ylim=(14, 26))
	# ax.set_title('scale = {}'.format(""), y=1)

	add_stat_annotation(
		ax,
		data=data,
		x="Grammaticality",
		y="Probability",
		width=0.2,
		# box_pairs=[("ejective-ejective", "legal"),
		#            ("plain-ejective", "legal")],
		# box_pairs=[("stop-aspirate", "legal"),
		#            ("stop-ejective", "legal"),
		#            ("stop-ejective", "stop-aspirate")],
		box_pairs=[("grammatical", "ungrammatical"),
				#    ("illegal-ejective", "illegal-aspirate")
					],
		# test='Kruskal',
		test='Mann-Whitney-ls',
		# test='Levene',
		text_format='full',
		loc='inside',
		verbose=1)

	plt.savefig('Judgement.png', bbox_inches="tight", dpi=600)  # Option 3

	# plt.show()
	
if __name__ == '__main__':
	# JudgementFile = 'data\\TurkishJudgement-MaxEnt-MaxOE1.txt'
	# JudgementFile = 'data\\TurkishJudgement-categorical.txt'
	# JudgementFile = 'data\\TurkishJudgement-probabilistic-100ep005lr.txt'
	# JudgementFile = 'data\\TurkishJudgementFile-tolerance.txt'
	# visual_judgefile_maxent(JudgementFile)


	humanJudgement = "data\\Daland_etal_2011__AverageScores.csv"
	visual_judgefile(humanJudgement)
