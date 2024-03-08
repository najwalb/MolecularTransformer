import pandas as pd
from rdkit import Chem
import numpy as np

def canonicalize(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return np.nan
    return Chem.MolToSmiles(m)

path1 = '/scratch/project_2006950/MolecularTransformer/experiments/results/7ckmnkvc_eval/eval_epoch280_steps100_resorted_0.9_cond4992_sampercond100_val_lam0.9_RB_translated_reprocTrue_keepTrue_chargesTrue_parsed.txt'
path2 = '/scratch/project_2006950/MolecularTransformer/experiments/results/7ckmnkvc_eval/eval_epoch360_steps100_resorted_0.9_cond4992_sampercond100_val_lam0.9_RB_translated_reprocTrue_keepTrue_chargesTrue_parsed.txt'
path2 = '/scratch/project_2006950/MolecularTransformer/experiments/results/82wscv5d_eval/eval_epoch280_steps100_resorted_0.9_cond4992_sampercond100_val_lam0.9_RB_translated_reprocTrue_keepTrue_chargesTrue_parsed.txt'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

# print(f"df1['pred'].head().sort_values() {df1['pred'].sort_values().reset_index(drop=True).head()}\n")
# print(f"df2['pred'].head().sort_values() {df2['pred'].sort_values().reset_index(drop=True).head()}\n")

assert (df1['product'].sort_values().reset_index(drop=True)==df2['product'].sort_values().reset_index(drop=True)).all(), 'Products are different'
assert (df1['pred'].apply(canonicalize).sort_values().reset_index(drop=True)!=df2['pred'].apply(canonicalize).sort_values().reset_index(drop=True)).any(), 'pred rcts are the same!'
assert (df1['pred_product'].apply(canonicalize).sort_values().reset_index(drop=True)!=df2['pred_product'].apply(canonicalize).sort_values().reset_index(drop=True)).any(), 'pred_product are the same!'