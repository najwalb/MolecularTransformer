'''
- Check reprocessing for true product/reactants
- compare products in own samples to retrobridge samples
- count unique predicted reactants per product

'''
import pandas as pd

path = '/Users/laabidn1/MolecularTransformer/data/retrobridge/retrobridge_samples.csv'
df = pd.read_csv(path)
nb_products = df.groupby('product').size().reset_index(name='counts')
print(f'Number of unique products: {len(nb_products)}')