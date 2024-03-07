from rdkit import Chem

my_smi = 'O=CC1=CN=CN1C1CC2=CC=CC=C2C1'
rb_smi = 'CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1'

out_smi = Chem.MolToSmiles(Chem.MolFromSmiles(my_smi))
print(out_smi)