import sys
import csv
import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import readwrite as rw
from bilinear import WordVectors, MarginalLogLinear, ConditionalSoftmax, ConditionalLogBilinear

def heatmap(model_filename):
	
	model = torch.load(model_filename)

	A_matrix = model.bilinear.weight[0].detach().numpy()
	print(A_matrix)
	# plt.imshow(A_matrix, cmap='hot', interpolation='nearest')
	# plt.show()

	s1 = ['+syll','-syll','+cons','-cons','+son','-son','+cont','-cont','+del rel','-del rel','+appr','-appr','+nas','-nas','+voi','-voi','+sp glo','-sp glo','+labi','-labi','+rd','-rd','+labiodent','-labiodent','+cor','-cor','+ant','-ant','+dist','-dist','+str','-str','+later','-later','+dors','-dors','+high','-high','+low','-low','+front','-front','+back','-back', '+tense', '-tense']
	s2 = ['+syll','-syll','+cons','-cons','+son','-son','+cont','-cont','+del rel','-del rel','+appr','-appr','+nas','-nas','+voi','-voi','+sp glo','-sp glo','+labi','-labi','+rd','-rd','+labiodent','-labiodent','+cor','-cor','+ant','-ant','+dist','-dist','+str','-str','+later','-later','+dors','-dors','+high','-high','+low','-low','+front','-front','+back','-back', '+tense', '-tense']

	fig, ax = plt.subplots()
	im = ax.imshow(A_matrix)

	# Show l ticks and label them with the respective list entries
	ax.set_xticks(np.arange(len(s1)), labels=s1)
	ax.set_yticks(np.arange(len(s2)), labels=s2)

	# Rotate the tick labels and set their ignment.
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
			rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	# for i in range(len(s1)):
	# 	for j in range(len(s2)):
	# 		text = ax.text(j, i, A_matrix[i, j],
	# 					ha="center", va="center", color="w")

	ax.set_title("A_matrix")
	fig.tight_layout()
	plt.show()
	print(model.w_linear.weight)
if __name__ == '__main__':
	heatmap(*sys.argv[1:])
	