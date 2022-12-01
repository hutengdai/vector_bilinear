import sys
import csv
import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import readwrite as rw
from bilinear import WordVectors, MarginalLogLinear, ConditionalSoftmax, ConditionalLogBilinear

def filter_matrix(matrix, labels, subset_labels):
	indices = [labels.index(x) for x in subset_labels]
	matrix = matrix.take(indices, axis=0).take(indices, axis=1)
	# breakpoint()
	return matrix

def heatmap(model_filename):
	
	model = torch.load(model_filename)

	A_matrix = model.bilinear.weight[0].detach().numpy()
	# plt.imshow(A_matrix, cmap='hot', interpolation='nearest')
	# plt.show()

	s1 = ['+syll','-syll','+cons','-cons','+son','-son','+cont','-cont','+del rel','-del rel','+appr','-appr','+nas','-nas','+voi','-voi','+sp glo','-sp glo','+labi','-labi','+rd','-rd','+labiodent','-labiodent','+cor','-cor','+ant','-ant','+dist','-dist','+str','-str','+later','-later','+dors','-dors','+high','-high','+low','-low','+front','-front','+back','-back', '+tense', '-tense']
	s2 = ['+syll','-syll','+cons','-cons','+son','-son','+cont','-cont','+del rel','-del rel','+appr','-appr','+nas','-nas','+voi','-voi','+sp glo','-sp glo','+labi','-labi','+rd','-rd','+labiodent','-labiodent','+cor','-cor','+ant','-ant','+dist','-dist','+str','-str','+later','-later','+dors','-dors','+high','-high','+low','-low','+front','-front','+back','-back', '+tense', '-tense']
	subset_labels = ['+cons','-cons','+son','-son','+cont','-cont','+del rel','-del rel','+appr','-appr']
	

	
	s1 = TH-_ W-_ S-_ T-_ V-_ HH-_ D-_ R-_ N-_ K-_ F-_ CH-_ B-_ L-_ DH-_ JH-_ G-_ P-_ Z-_ M-_ Y-_ SH-_

	s2 =_-TH _-W _-# _-S _-T _-V _-HH _-D _-R _-N _-K _-F _-CH _-B _-L _-DH _-JH _-G _-P _-Z _-M _-Y _-SH #-_ 
	
	A_matrix = filter_matrix(A_matrix, s1, subset_labels)
	print(A_matrix)

	fig, ax = plt.subplots()
	im = ax.imshow(A_matrix, cmap="seismic")

	# Show l ticks and label them with the respective list entries
	ax.set_xticks(np.arange(len(subset_labels)), labels=subset_labels)
	ax.set_yticks(np.arange(len(subset_labels)), labels=subset_labels)

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
	