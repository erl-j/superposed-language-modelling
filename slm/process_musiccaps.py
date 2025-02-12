import pandas as pd

df = pd.read_csv('data/musiccaps-public.csv')

# add column for original index, put first
df.insert(0, 'original_index', df.index)

# remove everything except original index and caption
df = df[['original_index', 'caption']]

# shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# export shuffled dataframe
df.to_csv('data/musiccaps-public-captions-shuffled.csv', index=False)

# keep only yid, 
                 