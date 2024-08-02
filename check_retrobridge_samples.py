'''
- Check reprocessing for true product/reactants
- compare products in own samples to retrobridge samples
- count unique predicted reactants per product

'''
import pandas as pd
from rdkit import Chem
import numpy as np

def canonicalize(smi):
    #m = Chem.MolFromSmiles(smi)
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return np.nan
    return Chem.MolToSmiles(m)

path = '/Users/laabidn1/MolecularTransformer/data/retrobridge/retrobridge_samples.csv'
df = pd.read_csv(path)
# df['prod_cano'] = df['product'].apply(canonicalize)

# get unique products from retrobridge samples
col = 'true'
unique = df[col].unique()
unique_cano = [canonicalize(p) for p in unique]
print(f'unique {col}: {set(unique.tolist())==set(unique_cano)}\n')
print(f'Number of unique {col}: {len(unique)}')
# print number of cano products that are nan
print(f'Number of unique cano {col} that are nan: {len([p for p in unique_cano if p is np.nan])}')

# check if prod_cano has nan
# print(df[df['prod_cano'].isna()])

# nb_products = df.groupby('product').size().reset_index(name='counts')
# print(f'Number of unique products: {len(nb_products)}')