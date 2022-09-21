import pandas as pd

if __name__ == '__main__':
	filepath = "hyperparameters"
	
	data = pd.read_csv(
			filepath,
			sep="\t",
			# header=0,
			encoding="utf-8")
		# shuffle
		data.sample(frac=1)

		pp.pprint(data)		