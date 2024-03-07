import argparse
import re
from pathlib import Path
import sys
import pandas as pd
from time import time
import os
import logging
import wandb
import pathlib
from rdkit import Chem
from functools import partial
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[0]

import re

"""
Chen, Shuan, and Yousung Jung.
"Deep retrosynthetic reaction prediction using local reactivity and global attention."
JACS Au 1.10 (2021): 1612-1620.

Predicted precursors are considered correct if 
- the predicted precursors are the same as the ground truth
- Molecular Transformer predicts the target product for the proposed precursors
"""
'''
python3 retrobridge_like_roundtrip.py --csv_out testing_wandb.out --wandb_run_id bznufq64 --n_conditions 4992 --steps 250 --epochs 200 --edge_conditional_set test --n_samples_per_condition 100 --reprocess_like_retrobridge True --keep_unprocessed True --remove_charges True
'''

def remove_charges_func(reactant):
    # Regular expression to match charges (+, -, and numerical charges like +2, -3, etc.)
    charge_pattern = re.compile(r'(\d+)?[\+\-]')
    # Remove charges from the reactant string
    cleaned_reactant = re.sub(charge_pattern, '', reactant)
    # print(reactant, cleaned_reactant)
    return cleaned_reactant

def read_saved_reaction_data_like_retrobridge(output_file, remove_charges=False, reprocess_like_retrobridge=False, keep_unprocessed=False):
    # Reads the saved reaction data from the samples.txt file
    # Split the data into individual blocks based on '(cond ?)' pattern
    data = open(output_file, 'r').read()
    blocks = re.split(r'\(cond \d+\)', data)[1:]
    reactant_predictions = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        original_reaction = lines[0].split(':')[0].strip()
        prod = original_reaction.split('>>')[-1]
        if reprocess_like_retrobridge: prod = Chem.MolToSmiles(Chem.MolFromSmiles(prod))
        if remove_charges: prod = remove_charges_func(prod)
        true_reactants = original_reaction.split('>>')[0]
        if reprocess_like_retrobridge: 
            if '.' in true_reactants:
                true_reactants = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(rct)) for rct in true_reactants.split('.')])
            else:
                true_reactants = Chem.MolToSmiles(Chem.MolFromSmiles(true_reactants))
        if remove_charges: true_reactants = remove_charges_func(true_reactants)
        for line in lines[1:]:
            match = re.match(r"\t\('([^']+)', \[([^\]]+)\]\)", line)
            if match:
                reaction_smiles = match.group(1)
                orig_pred = reaction_smiles.split('>>')[0]
                if reprocess_like_retrobridge: 
                    try:
                        if '.' in pred_reactants:
                            pred_reactants = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(rct)) for rct in orig_pred.split('.')])
                        else:
                            mol = Chem.MolFromSmiles(orig_pred)
                            if mol: pred_reactants = Chem.MolToSmiles(mol)
                            else: pred_reactants = orig_pred if keep_unprocessed else ''
                    except:
                        pred_reactants = orig_pred if keep_unprocessed else ''
                if remove_charges: orig_pred = remove_charges_func(orig_pred)
                # numbers = list(map(float, match.group(2).split(',')))
                # generated_reactions.append((reaction_smiles, numbers))
                # maybe don't care about the numbers for now?
                reactant_predictions.append((prod, true_reactants, pred_reactants))

    return reactant_predictions

def get_artifact_with_newest_alias(wandb_entity, wandb_project, wandb_run_id, collection_name, epoch, 
                                   steps, n_conditions, n_samples_per_condition, edge_conditional_set):
    collection_name = f"{wandb_run_id}_eval"
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name='eval', project=f'{wandb_entity}/{wandb_project}').collections()
        if coll.name==collection_name
    ]
    assert len(collections)==1, f'Found {len(collections)} collections with name {collection_name}, expected 1.'
    
    coll = collections[0]
    ## TODO: might want to agree on a format of the alias and update the code below
    ## TODO: test this once good file is on wandb

    aliases = [alias for art in coll.versions() for alias in art.aliases \
                     if ('eval' in alias and 'cond' in alias and 'steps' in alias) \
                     and re.findall('epoch\d+', alias)[0]==f'epoch{epoch}' \
                     and re.findall('steps\d+', alias)[0]==f'steps{steps}' \
                     and re.findall('cond\d+', alias)[0]==f'cond{n_conditions}' \
                     and re.findall('sampercond\d+', alias)[0]==f'sampercond{n_samples_per_condition}' \
                     and alias.split('_')[7].split('.txt')[-1]==edge_conditional_set]
    
    versions = [int(art.version.split('v')[-1]) for art in coll.versions()]

    aliases = [a for a,v in sorted(zip(aliases, versions), key=lambda pair: -pair[1])]
    
    assert len(aliases)>0, f'No alias found.'
   
    return aliases[0]
 
def donwload_eval_file_from_artifact(wandb_entity, wandb_project, wandb_run_id, epoch, steps, n_conditions, n_samples_per_condition, edge_conditional_set, savedir):
    collection_name = f"{wandb_run_id}_eval"
    alias = get_artifact_with_newest_alias(wandb_entity, wandb_project, wandb_run_id, collection_name, 
                                           epoch, steps, n_conditions, n_samples_per_condition, edge_conditional_set)
    
    artifact_name = f"{wandb_entity}/{wandb_project}/{collection_name}:{alias}"
    samples_art = wandb.Api().artifact(artifact_name)
    samples_art.download(root=savedir)
    
    return os.path.join(savedir, f"{alias}.txt"), artifact_name

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    https://github.com/pschwllr/MolecularTransformer/tree/master#pre-processing
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    if smi != ''.join(tokens):
        print(smi, ''.join(tokens))
    return ' '.join(tokens)

def _assign_groups(df, samples_per_product):
    # group is the number of conditioning product
    # for test data: group is [0,5006], with each 10 samples having the same index
    df['group'] = np.arange(len(df)) // samples_per_product
    return df

def assign_groups(df, samples_per_product_per_file=10):
    # what does partial do here?
    # it's a way to fix a function's arguments to a specific value
    # what happens if we group by from_file then by group?
    # for each file, we assign numbers 0...nb_product_in_subset to each group of 10 samples 
    # (i.e. assign the index of the product to each group of 10 samples)
    df = df.groupby('from_file', group_keys=False).apply(partial(_assign_groups, samples_per_product=samples_per_product_per_file))
    return df

def compute_confidence(df, group_key='product'):
    # count the number of times a prediction is made for a given product/group => repetitions!!
    counts = df.groupby([group_key, 'pred'], group_keys=False).size().reset_index(name='count') 
    # group = single product => count the number of predictions we get per product => nb_samples_per_product
    group_size = df.groupby([group_key], group_keys=False).size().reset_index(name='group_size') # shld be 100 in paper experiments

    #     # Don't use .merge() as it can change the order of rows
    ##     df = df.merge(counts, on=['group', 'pred'], how='left')
    #     df = df.merge(counts, on=['group', 'pred'], how='inner')
    #     df = df.merge(group_size, on=['group'], how='left')
    
    counts_dict = {(g, p): c for g, p, c in zip(counts[group_key], counts['pred'], counts['count'])}
    df['count'] = df.apply(lambda x: counts_dict[(x[group_key], x['pred'])], axis=1)

    size_dict = {g: s for g, s in zip(group_size[group_key], group_size['group_size'])}
    df['group_size'] = df.apply(lambda x: size_dict[x[group_key]], axis=1)

    # how many times the same prediction is made for a given product
    df['confidence'] = df['count'] / df['group_size']

    # sanity check
    # what does nunique do?
    # it counts the number of unique values in a series: all product_pred have the same confidence value
    # all groups have the same group_size (n_samples is the same for all products)
    assert (df.groupby([group_key, 'pred'], group_keys=False)['confidence'].nunique() == 1).all()
    assert (df.groupby([group_key], group_keys=False)['group_size'].nunique() == 1).all()

    return df

def get_top_k(df, k, scoring=None):
    if callable(scoring):
        df["_new_score"] = scoring(df)
        scoring = "_new_score"

    if scoring is not None:
        df = df.sort_values(by=scoring, ascending=False)
        
    # what does this do?
    # drop duplicates of the pred column
    df = df.drop_duplicates(subset='pred')

    return df.head(k)

def canonicalize(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return np.nan
    return Chem.MolToSmiles(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--csv_file", type=Path, required=True)
    parser.add_argument("--csv_out", type=Path, required=False, default=True)
    parser.add_argument("--mol_trans_dir", type=Path, default="./")
    parser.add_argument("--reprocess_like_retrobridge", type=bool, default=False)
    parser.add_argument("--keep_unprocessed", type=bool, default=False)
    parser.add_argument("--remove_charges", type=bool, default=False)
    # added these params to be able to get data from wandb
    parser.add_argument('--wandb_run_id', type=str, required=True,
                       help='The id of the training run on wandb')
    parser.add_argument('--n_conditions', type=str, required=True,
                       help='to identify the samples artifact')
    parser.add_argument('--epochs', nargs='+', type=int, required=True,
                       help='epochs')
    parser.add_argument('--steps', type=int, required=True,
                        help='sampling steps')
    parser.add_argument('--round_trip_k', nargs='+', type=int, default=[1, 3, 5, 10, 50],
                       help='list of k values for the round_trip accuracy')
    parser.add_argument('--n_samples_per_condition', type=str, default="100",
                       help='nb samples per condition')
    parser.add_argument('--edge_conditional_set', type=str, default='test',
                       help='edge conditional set')
    parser.add_argument('--log_to_wandb', type=bool, default=False,
                       help='log_to_wandb')
    args = parser.parse_args()

    sys.path.append(str(args.mol_trans_dir))
    import onmt
    from onmt.translate.translator import build_translator
    import onmt.opts

    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    args = parser.parse_args(sys.argv[1:] + [
        "-model", str(Path(args.mol_trans_dir, 'experiments/models', 'MIT_mixed_augm_model_average_20.pt')),
        "-src", "input.txt", "-output", "pred.txt ",
        "-replace_unk", "-max_length", "200", "-fast"
    ])
    
    # 1. get the evaluation file from wandb
    wandb_entity = 'najwalb'
    wandb_project = 'retrodiffuser'
    model = "MIT_mixed_augm_model_average_20.pt"
    
    assert len(args.epochs)==1, 'The script can only handle one epoch for now.'
    epoch = args.epochs[0]
    log.info(f'Getting eval file from wandb...\n')
    savedir = os.path.join(parent_path, "data", f"{args.wandb_run_id}_eval")
    log.info(f'==== Saving artifact in {savedir}\n')
    eval_file, artifact_name = donwload_eval_file_from_artifact(wandb_entity, wandb_project, args.wandb_run_id, epoch, args.steps, args.n_conditions, 
                                                                args.n_samples_per_condition, args.edge_conditional_set, savedir)
    
    # 2. read saved reaction data from eval file
    log.info(f'Read saved reactions...\n')
    reactions = read_saved_reaction_data_like_retrobridge(eval_file, remove_charges=args.remove_charges, reprocess_like_retrobridge=args.reprocess_like_retrobridge, keep_unprocessed=args.keep_unprocessed)

    # 3. output reaction data to text file as one reaction per line
    # log.info(f'Output reactions to file for processing...\n')
    # mt_opts = f'RB_translated_reproc{args.reprocess_like_retrobridge}_keep{args.keep_unprocessed}_charges{args.remove_charges}'
    # eval_file_name = eval_file.split('/')[-1].split('.txt')[0]
    # dataset = f'{args.wandb_run_id}_eval' # use eval_run name (collection name) as dataset name
    # reaction_file_name = f"{eval_file_name}_{mt_opts}_parsed.txt" # file name is: alias name from wandb+translation options+parsed (to mark that it was parsed)
    # reaction_file_path = os.path.join(parent_path, 'data', dataset, reaction_file_name) 
    # log.info(f'==== reaction_file_path {reaction_file_path}\n')
    # open(reaction_file_path, 'w').write('product,true,pred\n')
    # open(reaction_file_path, 'a').writelines([f'{prod},{true},{pred}\n' for prod,true,pred in reactions])
    
    # Read CSV
    # turn list into df
    df = pd.read_csv(reaction_file_path)

    # Find unique SMILES
    unique_smiles = list(set(df['pred']))

    # Tokenize
    tokenized_smiles = [smi_tokenizer(s.strip()) for s in unique_smiles]

    print("Predicting products...")
    tic = time()
    translator = build_translator(args, report_score=True)
    scores, pred_products = translator.translate(
        src_data_iter=tokenized_smiles,
        batch_size=args.batch_size,
        attn_debug=args.attn_debug
    )
    # always take n_best = 1?
    pred_products = [x[0].strip() for x in pred_products]
    print("... done after {} seconds".format(time() - tic))

    # De-tokenize
    pred_products = [''.join(x.split()) for x in pred_products]

    # gather results
    pred_products = {r:p for r, p in zip(unique_smiles, pred_products)}

    # update dataframe
    df['pred_product'] = [pred_products[r] for r in df['pred']]

    # Write results
    if args.csv_out:
        print("Writing CSV file...")
        df.to_csv(os.path.join('experiments', 'results', dataset, reaction_file_name), index=False)

    # translation_out_file = args.translation_out_file
    # df_in = pd.read_csv(translation_out_file)
    # df['from_file'] = eval_file_name 
    
    # df = assign_groups(df, samples_per_product_per_file=10)
    # df.loc[(df['product'] == 'C') & (df['true'] == 'C'), 'true'] = 'Placeholder'
    # df = compute_confidence(df, group_key='product')
    # TODO: replace with our own count
    
    for key in ['product', 'pred_product']:
        df[key] = df[key].apply(canonicalize)

    df['exact_match'] = df['true'] == df['pred']
    df['round_trip_match'] = df['product'] == df['pred_product']
    df['match'] = df['exact_match'] | df['round_trip_match']
    
    avg = {}
    ranking_metric = 'confidence'
    for k in args.round_trip_k:
        topk_df = df.groupby(['product']).apply(partial(get_top_k, k=k, scoring=lambda df:np.log(df[ranking_metric]))).reset_index(drop=True)
        avg[f'roundtrip-coverage-{k}_weighted_0.9'] = topk_df.groupby('product').match.any().mean()
        avg[f'roundtrip-accuracy-{k}_weighted_0.9'] = topk_df.groupby('product').match.mean().mean()
    print(f'avg {avg}\n')

    # 8. log scores to wandb
    print(f'args.log_to_wandb {args.log_to_wandb}\n')
    if args.log_to_wandb:
        wandb_entity = 'najwalb'
        wandb_project = 'retrodiffuser'
        model = "MIT_mixed_augm_model_average_20.pt"
        run = wandb.init(name=f'{wandb_run_id}_{reaction_file_name}', project=wandb_project, entity=wandb_entity, resume='allow', job_type='round_trip', 
                         config={"experiment_group": 'retrobridge_samples'})
        run.log({'round_trip_k/': avg})