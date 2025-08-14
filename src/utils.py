

def get_subset(df, subset):
	return df[df['filename_index'].str.contains(subset, case=False, na=False)]