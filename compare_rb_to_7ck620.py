import pandas as pd
from rdkit import Chem
import re

def atom_count(smi):
    atoms = [a for a in Chem.MolFromSmiles(smi).GetAtoms()]
    
    return len(atoms)

def remove_charges_func(reactant):
    # Regular expression to match charges (+, -, and numerical charges like +2, -3, etc.)
    charge_pattern = re.compile(r'(\d+)?[\+\-]')
    # Remove charges from the reactant string
    cleaned_reactant = re.sub(charge_pattern, '', reactant)
    # print(reactant, cleaned_reactant)
    return cleaned_reactant

def reconstruct_mol_from_atoms_and_bonds_only(old_mol, return_smi=False):
    if type(old_mol)==str: old_mol = Chem.MolFromSmiles(old_mol)
    if old_mol is None: return None
    
    # create atoms based on labels only
    atoms = [a.GetSymbol() for a in old_mol.GetAtoms()]
    bonds = [(b.GetBeginAtom().GetIdx(),b.GetEndAtom().GetIdx(),b.GetBondType()) for b in old_mol.GetBonds()]
    new_mol = Chem.RWMol()
    for a in atoms: new_mol.AddAtom(Chem.Atom(a))
    for beg_atom_idx, end_atom_idx, bond_type in bonds: new_mol.AddBond(beg_atom_idx, end_atom_idx, bond_type)
    
    if return_smi: return Chem.MolToSmiles(new_mol)
    
    return new_mol

# rb samples as given by RB in their iclr submission on openreview
rb_samples_path = '/scratch/project_2006950/MolecularTransformer/data/retrobridge/retrobridge_samples_new.csv'
run_7ck620_samples_path = '/scratch/project_2006950/MolecularTransformer/experiments/results/None_eval/7ck_620_reprocessFalse_keepUnprocessedFalse_removeChargesFalse_parsed.txt'
run_7ck620_samples_nocharges_path = '/scratch/project_2006950/MolecularTransformer/experiments/results/None_eval/7ck_620_reprocessFalse_keepUnprocessedFalse_removeChargesTrue_parsed.txt'
uspto50k_test = '/scratch/project_2006950/MolecularTransformer/data/uspto50k/uspto50k_test.csv'
uspto50k_test_from_rb = '/scratch/project_2006950/MolecularTransformer/data/uspto50k/uspto50k_test_from_rb.csv'
uspto50k_test_without_am = '/scratch/project_2006950/MolecularTransformer/data/uspto50k/uspto50k_test_without_am.csv'
uspto50k_test_without_am_without_stereo = '/scratch/project_2006950/MolecularTransformer/data/uspto50k/uspto50k_test_without_am_without_stereo.csv'
uspto50k_test_without_am_without_stereo_without_charges = '/scratch/project_2006950/MolecularTransformer/data/uspto50k/uspto50k_test_without_am_without_stereo_without_charges.csv'
uspto50k_test_from_atoms_and_bonds = '/scratch/project_2006950/MolecularTransformer/data/uspto50k/uspto50k_test_from_atoms_and_bonds.csv'
uspto50k_test_from_rb_from_atoms_and_bonds = '/scratch/project_2006950/MolecularTransformer/data/uspto50k/uspto50k_test_from_rb_from_atoms_and_bonds.csv'

# look into the 55 products that are in rb test and not in our test set=> print the full reactions
#test_from_rb_reactions = [l.split(',')[-1] for l in open(uspto50k_test_from_rb, 'r').readlines()[1:]]
#test_from_rb_df = pd.DataFrame([[r.split('>>')[1], r.split('>>')[0], r] for r in test_from_rb_reactions], columns=['product', 'true', 'reaction'])
samples_from_rb_df = pd.read_csv(rb_samples_path)
samples_from_rb_df = samples_from_rb_df[samples_from_rb_df['product']!='C'] # we don't care about the placeholder product here
samples_from_7ck620_df = pd.read_csv(run_7ck620_samples_path)

# TODO: work with the samples we do have in common
samples_from_7ck620_df['product_reconstructed_from_atoms_and_bonds'] = samples_from_7ck620_df['product'].apply(lambda x:reconstruct_mol_from_atoms_and_bonds_only(x,return_smi=True))
common_products = set(samples_from_rb_df['product'].unique()).intersection(set(samples_from_7ck620_df['product_reconstructed_from_atoms_and_bonds'].unique()))
print(f'len(common_products) {len(common_products)}\n')
samples_from_rb_in_common = samples_from_rb_df[samples_from_rb_df['product'].isin(common_products)]
print(f'samples_from_rb_in_common {samples_from_rb_in_common["product"].nunique()}\n')
samples_from_rb_in_common.to_csv('/scratch/project_2006950/MolecularTransformer/experiments/results/None_eval/samples_from_rb_in_common.csv')
samples_from_7ck620_in_common = samples_from_7ck620_df[samples_from_7ck620_df['product_reconstructed_from_atoms_and_bonds'].isin(common_products)]
print(f'samples_from_7ck620_in_common {samples_from_7ck620_in_common["product_reconstructed_from_atoms_and_bonds"].nunique()}\n')
samples_from_7ck620_in_common.to_csv('/scratch/project_2006950/MolecularTransformer/experiments/results/None_eval/samples_from_7ck620_in_common.csv')
exit()

# check that the products dropped in rb_samples is because the number of dummy nodes is not enough
# NOTE: there are still 5 products unaccounted for
test_from_rb_df['needed_n_dummy_nodes'] = test_from_rb_df['true'].apply(atom_count)-test_from_rb_df['product'].apply(atom_count)
print(f'# of unique products in samples_from_rb_df {samples_from_rb_df["product"].nunique()}\n')
print(f'# of unique products in test_from_rb_df {test_from_rb_df["product"].nunique()}\n')
print(f'needed_n_dummy_nodes {(test_from_rb_df["needed_n_dummy_nodes"]>10).sum()}\n')

print(f'# of unique products in test from rb {test_from_rb_df["product"].nunique()}\n')
test_reactions = open(uspto50k_test, 'r').readlines() # file contains reactions only
test_df = pd.DataFrame([[r.split('>>')[1], r.split('>>')[0], r] for r in test_reactions], columns=['product', 'true', 'reaction'])
print(f'# of unique products in test {test_df["product"].nunique()}\n')
# NOTE: difference below includes differences in atom-mapping only
products_in_rb_only = set(test_from_rb_df["product"]).difference(set(test_df["product"]))
reactions_in_rb_only = test_from_rb_df[~test_from_rb_df['product'].isin(test_df['product'])]
print(f'reactions in test not in rb_test {len(reactions_in_rb_only)}\n')

# TODO: save reactions in rb test not in ours in the style of rb samples
# samples_in_rb_only = samples_from_rb_df[~samples_from_rb_df['product']]

# check that rb_samples products are in our test set
test_df['product_reconstructed_from_atoms_and_bonds'] = test_df['product'].apply(lambda x: reconstruct_mol_from_atoms_and_bonds_only(x, return_smi=True))
test_from_rb_df['product_reconstructed_from_atoms_and_bonds'] = test_from_rb_df['product'].apply(lambda x: reconstruct_mol_from_atoms_and_bonds_only(x, return_smi=True))
rb_samples_products_in_test = samples_from_rb_df[samples_from_rb_df['product'].isin(test_df['product_reconstructed_from_atoms_and_bonds'])]
test_products_not_in_rb_samples = test_df[~test_df['product_reconstructed_from_atoms_and_bonds'].isin(samples_from_rb_df['product'])]
rb_samples_products_in_test_from_rb = samples_from_rb_df[samples_from_rb_df['product'].isin(test_from_rb_df['product_reconstructed_from_atoms_and_bonds'])]
print(f'products in rb_samples that are also in test {rb_samples_products_in_test["product"].nunique()}\n')
print(f'products in test that are not in rb_samples {test_products_not_in_rb_samples["product"].nunique()}\n')
print(f'products in rb_samples that are also in test_from_rb {rb_samples_products_in_test_from_rb["product"].nunique()}\n')
# NOTE: # of products in rb_samples = # of products in test_from_rb - (# of products dummy nodes > 10) - 5?
# NOTE: # of products in test = # of product in rb_samples - 55 missing products
# NOTE: prods in test_rb = prods in test + 55 (missing for some reason)



# remove atom-mapping from uspto-50k test and check the unique count

# lines = open(uspto50k_test_from_rb, 'r').readlines()
# reactions = [l.split(',')[-1] for l in lines[1:]]
# print(f'reactions with am {len(set(reactions))}\n')
# reactions_without_am = []
# reactions_without_am_without_stereo = []
# reactions_without_am_without_stereo_without_charges = [] mv
# reactions_from_atoms_and_bonds = []

# for r in reactions:
#     #print(f'r {r}\n')
#     rcts = r.strip().split('>>')[0]
#     prods = r.strip().split('>>')[-1]
    
#     rcts_mol = Chem.MolFromSmiles(rcts)
#     prods_mol = Chem.MolFromSmiles(prods)
    
#     # [a.ClearProp('molAtomMapNumber') for a in rcts_mol.GetAtoms()]
#     # rcts_without_am = Chem.MolToSmiles(rcts_mol)

#     # [a.ClearProp('molAtomMapNumber') for a in prods_mol.GetAtoms()]
#     # prods_without_am = Chem.MolToSmiles(prods_mol)
    
#     # reactions_without_am.append(rcts_without_am+'>>'+prods_without_am+'\n')
    
#     # Chem.RemoveStereochemistry(rcts_mol)
#     # Chem.RemoveStereochemistry(prods_mol)
#     # reactions_without_am_without_stereo.append(Chem.MolToSmiles(rcts_mol)+'>>'+Chem.MolToSmiles(prods_mol)+'\n')
    
#     # remove formal charge (with regex)
#     #reactions_without_am_without_stereo_without_charges.append(remove_charges_func(Chem.MolToSmiles(rcts_mol))+'>>'+remove_charges_func(Chem.MolToSmiles(prods_mol))+'\n')
    
#     new_rcts_smi = Chem.MolToSmiles(get_rw_mol_from_atoms_and_bonds(rcts_mol))
#     new_prods_smi = Chem.MolToSmiles(get_rw_mol_from_atoms_and_bonds(prods_mol))
#     reactions_from_atoms_and_bonds.append(new_rcts_smi+'>>'+new_prods_smi+'\n')

# print(f'reactions_from_atoms_and_bonds set {len(set(reactions_from_atoms_and_bonds))}\n')
# open(uspto50k_test_from_rb_from_atoms_and_bonds,'w').writelines(reactions_from_atoms_and_bonds)
# exit()

# print(f'reactions_without_am {len(set(reactions_without_am))}\n')
# print(f'reactions_without_am_without_stereo {len(set(reactions_without_am_without_stereo))}\n')
# print(f'reactions_without_am_without_stereo_without_charges {len(set(reactions_without_am_without_stereo_without_charges))}\n')

# open(uspto50k_test_without_am,'w').writelines(reactions_without_am)
# open(uspto50k_test_without_am_without_stereo,'w').writelines(reactions_without_am_without_stereo)
# open(uspto50k_test_without_am_without_stereo_without_charges,'w').writelines(reactions_without_am_without_stereo_without_charges)

# rb_reactions = open(uspto50k_test_from_rb_from_atoms_and_bonds,'r').readlines()
# print(f'len(rb_reactions) {len(rb_reactions)}\n')
# test_reactions = open(uspto50k_test_from_atoms_and_bonds,'r').readlines()
# print(f'len(test_reactions) {len(test_reactions)}\n')

lines = open(uspto50k_test_from_rb,'r').readlines()
rb_reactions = [l.split(',')[-1] for l in lines[1:]]
rb_reactions_list = []
for r in rb_reactions:
    # prod, pred
    rb_reactions_list.append([r.strip().split('>>')[-1], r.strip().split('>>')[0]])
    
rb_reactions_df = pd.DataFrame(rb_reactions_list, columns=['product', 'pred'])
print(f'# of unique products in rb original test set {len(set(rb_reactions_df["product"]))}\n')
print(f'rb_reactions_df multi product {len([p for p in rb_reactions_df["product"].values if "." in p])}')


test_reactions_list = []
for r in test_reactions:
    # prod, pred
    test_reactions_list.append([r.strip().split('>>')[-1], r.strip().split('>>')[0]])
    
test_reactions_df = pd.DataFrame(test_reactions_list, columns=['product', 'pred'])
rb_samples_df = pd.read_csv(rb_samples_path)

print(f'rb_reactions_df shape {rb_reactions_df.groupby("product").agg({"pred":"size"}).shape}\n')
print(f'test_reactions_df shape {test_reactions_df.groupby("product").agg({"pred":"size"}).shape}\n')

# get a list of the duplicated products

in_rb_not_in_test = set(rb_reactions_df["product"]).difference(set(test_reactions_df["product"]))
in_test_not_in_rb = set(test_reactions_df["product"]).difference(set(rb_reactions_df["product"]))
in_rb_samples_not_in_test = set(rb_samples_df["product"]).difference(set(test_reactions_df["product"]))
print(f'in_rb_not_in_test {len(in_rb_not_in_test)}\n')
print(f'in_test_not_in_rb {len(in_test_not_in_rb)}\n')
print(f'in_rb_samples_not_in_test {len(in_rb_samples_not_in_test)}\n')
exit()

# rb_df = pd.read_csv(rb_samples_path)
# # run_7ck620_df = pd.read_csv(run_7ck620_samples_path)
# # run_7ck620_nocharges_df = pd.read_csv(run_7ck620_samples_nocharges_path)

# group_by_product = rb_df.groupby('product').agg({'true':'size'})
# print(f'rb {group_by_product.shape}\n')
# print(f'common prods {len(list(set(rb_df["product"]) & set(reactions_df["product"])))}\n')
# in_rb_not_in_test = set(rb_df["product"]).difference(set(reactions_df["product"]))
# in_test_not_in_rb = set(reactions_df["product"]).difference(set(rb_df["product"]))
# print(f'in_rb_not_in_test {len(in_rb_not_in_test)}\n')
# print(f'in_test_not_in_rb {len(in_test_not_in_rb)}\n')

# print(f'example diff {rb_df[rb_df["product"]=="C"]["true"]}\n')

# for the products in common, save in a file and compare accuracy just on those


# print(f'common diff {len(list(set(reactions_df["product"]).difference(set(rb_df["product"]))))}\n')
# print(f"7ck {run_7ck620_df.groupby('product').agg({'true':'size'}).shape}\n")
# print(f"7ck no charges {run_7ck620_nocharges_df.groupby('product').agg({'true':'size'}).shape}\n")

# rb_df = pd.DataFrame(, columns=["product", "pred", "true", "score", "true_n_dummy_nodes", "sampled_n_dummy_nodes", "nll", "ell", "from_file,pred_product"])