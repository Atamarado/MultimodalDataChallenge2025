import pandas as pd
import numpy as np

df = pd.read_csv('data/metadata.csv')

def generate_temporal_embeddings(df):
	df['eventDate'] = pd.to_datetime(df['eventDate'])

	df['month_sin'] = np.sin(2 * np.pi * df['eventDate'].dt.month / 12)
	df['month_cos'] = np.cos(2 * np.pi * df['eventDate'].dt.month / 12)
	df['day_of_year_sin'] = np.sin(2 * np.pi * df['eventDate'].dt.dayofyear / 365.25)
	df['day_of_year_cos'] = np.cos(2 * np.pi * df['eventDate'].dt.dayofyear / 365.25)
	df['week_sin'] = np.sin(2 * np.pi * df['eventDate'].dt.isocalendar()['week'] / 52)
	df['week_cos'] = np.cos(2 * np.pi * df['eventDate'].dt.isocalendar()['week'] / 52)

	max_year = 2020
	min_year = 1985
	df['year'] = 2 * (df['eventDate'].dt.year - min_year) / (max_year - min_year) - 1

	return df

df = generate_temporal_embeddings(df)

def correlation_ratio(categories, measurements):
	"""
	Compute correlation ratio (η) for categorical-numeric association.
	"""
	# Remove NaN pairs
	mask = pd.notna(categories) & pd.notna(measurements)
	categories = np.array(categories[mask])
	measurements = np.array(measurements[mask])

	if len(np.unique(categories)) <= 1 or len(np.unique(measurements)) <= 1:
		return 0  # No variation → no correlation

	category_means = [measurements[categories == cat].mean() for cat in np.unique(categories)]
	grand_mean = measurements.mean()

	ss_between = sum([
		len(measurements[categories == cat]) * (mean - grand_mean) ** 2
		for cat, mean in zip(np.unique(categories), category_means)
	])

	ss_total = sum((measurements - grand_mean) ** 2)

	return np.sqrt(ss_between / ss_total) if ss_total != 0 else 0

from utils import get_subset
df = get_subset(df, 'train')

temporal_columns = ['month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'week_sin', 'week_cos', 'year']
results = {col: correlation_ratio(df["taxonID_index"], df[col]) for col in temporal_columns}

correlation_df = pd.DataFrame.from_dict(results, orient='index', columns=["Correlation Ratio (η)"])

pass