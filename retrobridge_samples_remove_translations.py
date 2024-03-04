import pandas as pd

samples_path = '/scratch/project_2006950/MolecularTransformer/data/retrobridge/restrobridge_samples.csv'
samples_retranslated_path = '/scratch/project_2006950/MolecularTransformer/data/retrobridge/retrobridge_samples_retranslated.csv'

df_orig = pd.read_csv(samples_path)
df_retrans = pd.read_csv(samples_retranslated_path)
df_retrans['from_file'] = df_orig['from_file']
df_retrans.to_csv(samples_retranslated_path, index=False)

# samples_without_translation_path = '/scratch/project_2006950/MolecularTransformer/data/retrobridge/retrobridge_samples_without_translation.csv'
# df = pd.read_csv(samples_path)
# df[['product', 'true', 'pred', 'from_file']].to_csv(samples_without_translation_path, index=False)
