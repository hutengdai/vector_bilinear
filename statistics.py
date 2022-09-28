import pandas as pd
import torch
if __name__ == '__main__':
	filepath = "hyperparameters"
	for files in filepath
	data = pd.read_csv(
			filepath,
			sep="\t",
			# header=0,
			encoding="utf-8")
		# shuffle
	data.sample(frac=1)

	print(data)
	torch.eval()