import pandas as pd

samples_path = '/Users/laabidn1/MolecularTransformer/data/retrobridge/retrobridge_samples.csv'
samples_without_translation_path = '/Users/laabidn1/MolecularTransformer/data/retrobridge/retrobridge_samples_without_translation.csv'

df = pd.read_csv(samples_path)
df[['product', 'true', 'pred']].to_csv(samples_without_translation_path)
