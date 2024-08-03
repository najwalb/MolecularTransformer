'''
    Compute topk round trip scores after generating products through translation
'''
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
import numpy as np
from functools import partial

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[0]

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

# counting! => doesn't use elbo at all ...
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=False, default=True)
    parser.add_argument('--round_trip_k', nargs='+', type=int, default=[1, 3, 5, 10, 50],
                       help='list of k values for the round_trip accuracy')
    parser.add_argument('--log_to_wandb', action='store_true', default=False, help='log_to_wandb')
    args = parser.parse_args()

    input_file = args.input_file
    df = pd.read_csv(input_file)
    
    df = assign_groups(df, samples_per_product_per_file=10)
    
    group_key = 'group'
    df.loc[(df['product'] == 'C') & (df['true'] == 'C'), 'true'] = 'Placeholder'
    df = compute_confidence(df, group_key=group_key)
    
    for key in ['product', 'pred_product']:
        df[key] = df[key].apply(canonicalize)

    df['exact_match'] = df['true'] == df['pred']
    df['round_trip_match'] = df['product'] == df['pred_product']
    df['match'] = df['exact_match'] | df['round_trip_match']
    
    avg = {}
    for k in args.round_trip_k:
        topk_df = df.groupby([group_key]).apply(partial(get_top_k, k=k, scoring=lambda df:np.log(df['confidence']))).reset_index(drop=True)
        avg[f'roundtrip-coverage-{k}_weighted_0.9'] = topk_df.groupby(group_key).match.any().mean()
        avg[f'roundtrip-accuracy-{k}_weighted_0.9'] = topk_df.groupby(group_key).match.mean().mean()
    
    coverage_to_print = {k: f"{avg[f'roundtrip-coverage-{k}_weighted_0.9']}\n" for k in args.round_trip_k}
    accuracy_to_print = {k: f"{avg[f'roundtrip-accuracy-{k}_weighted_0.9']}\n" for k in args.round_trip_k}
    print(f'coverage:\n\t{coverage_to_print}\n')
    print(f'accuracy:\n\t{accuracy_to_print}')

    # 8. log scores to wandb
    reaction_file_name = args.input_file
    wandb_run_id = 'test'
    print(f'args.log_to_wandb {args.log_to_wandb}\n')
    if args.log_to_wandb:
        wandb_entity = 'najwalb'
        wandb_project = 'retrodiffuser'
        model = "MIT_mixed_augm_model_average_20.pt"
        run = wandb.init(name=f'{wandb_run_id}_{reaction_file_name}', project=wandb_project, entity=wandb_entity, resume='allow', job_type='round_trip', 
                         config={"experiment_group": 'retrobridge_samples'})
        run.log({'round_trip_k/': avg})
