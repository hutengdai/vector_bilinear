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

def heatmap(model_filename, tie):
	
	model = torch.load(model_filename)

	A_matrix = model.bilinear.weight[0].detach().numpy()
	# plt.imshow(A_matrix, cmap='hot', interpolation='nearest')
	# plt.show()

	full_labels = ['syll','cons','son','cont','del rel','appr','nas','voi','sp glo','labi','rd','labiodent','cor','ant','dist','str','later','dors','high','low','front','back','tense']
	labels = ['cons','son','cont','del rel','appr']
	
	if not int(tie):
		full_labels = [y for x in full_labels for y in ('+' + x, '-' + x)]

		labels = [y for x in labels for y in ('+' + x, '-' + x)]

	# tied labels mean the weights for negative features and postive features are the same
	 
	# s1 = TH-_ W-_ S-_ T-_ V-_ HH-_ D-_ R-_ N-_ K-_ F-_ CH-_ B-_ L-_ DH-_ JH-_ G-_ P-_ Z-_ M-_ Y-_ SH-_

	# s2 =_-TH _-W _-# _-S _-T _-V _-HH _-D _-R _-N _-K _-F _-CH _-B _-L _-DH _-JH _-G _-P _-Z _-M _-Y _-SH #-_ 
	
	A_matrix = filter_matrix(A_matrix, full_labels, labels)
	print(A_matrix)

	fig, ax = plt.subplots()
	im = ax.imshow(A_matrix, cmap="seismic")

	# Show l ticks and label them with the respective list entries
	ax.set_xticks(np.arange(len(labels)), labels=labels)
	ax.set_yticks(np.arange(len(labels)), labels=labels)

	# Rotate the tick labels and set their ignment.
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
			rotation_mode="anchor")

	ax.set_title("A_matrix")
	fig.tight_layout()
	# plt.show()

	print(model.w_linear.weight)
	plt.savefig('heatmap_%s.png' %str(model_filename).split(".")[0].replace("result/",""), dpi=300)  

	
if __name__ == '__main__':
	heatmap(*sys.argv[1:])
	