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
pass