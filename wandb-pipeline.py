#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Tokenize data
"""
import argparse
import onmt.opts
import os
import pathlib
import subprocess
import wandb
import re
import pandas as pd

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[0]

def extract_epoch_number(s):
    match = re.search(r"epoch(\d+)", s)
    if match:
        return int(match.group(1))
    else:
        return -1
    
def read_saved_reaction_data(output_file):
    # Reads the saved reaction data from the samples.txt file
    # Split the data into individual blocks based on '(cond ?)' pattern
    data = open(output_file, 'r').read()
    blocks = re.split(r'\(cond \d+\)', data)[1:]
    reactions = []
    for block in blocks:
        lines = block.strip().split('\n')
        original_reaction = lines[0].split(':')[0].strip()
        generated_reactions = []
        for line in lines[1:]:
            match = re.match(r"\t\('([^']+)', \[([^\]]+)\]\)", line)
            if match:
                reaction_smiles = match.group(1)
                # numbers = list(map(float, match.group(2).split(',')))
                # generated_reactions.append((reaction_smiles, numbers))
                # maybe don't care about the numbers for now?
                generated_reactions.append(reaction_smiles)
        reactions.append((original_reaction, generated_reactions))

    return reactions

def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0

def main(opt):
    # 0. get the evaluation file from wandb
    wandb_entity = 'najwalb'
    wandb_project = 'retrodiffuser'
    wandb_run_id = opt.wandb_eval_run_id
    n_conditions = opt.n_conditions
    epoch = opt.epoch
     
    collection_name = f"{wandb_run_id}_eval"
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name='eval', project=wandb_project).collections()
        if coll.name==collection_name
    ]
    assert len(collections)==1, f'Found {len(collections)} collections with name {collection_name}, expected 1.'
    
    coll = collections[0]
    ## TODO: might want to agree on a format of the alias and update the code below
    ## TODO: test this once good file is on wandb
    aliases = [alias for art in coll.versions() for alias in art.aliases \
                     if 'eval' in alias
                     and re.findall('epoch\d+', alias)[0]==f'epoch{epoch}'
                     and alias.split('_')[4]==n_conditions]
    versions = [int(art.version.split('v')[-1]) for art in coll.versions()]
    aliases = [a for a,v in sorted(zip(aliases, versions), key=lambda pair: -pair[1])]
    aliases[0] = "eval_epoch280_resorted_0.9_s128"
  
    # savedir = os.path.join(parent_path, 'data', collection_name)
    output_file = f'/Users/laabidn1/MolecularTransformer/data/50k_separated/{aliases[0]}.txt'
    
    # 0.1. parse the wandb file: to get 
    # => 1 point if: exact matches, round trip matches 
    # => round trip top k: if top k has a score 1
    #   => keep in same order as given in elbo
    reactions = read_saved_reaction_data(output_file)
    # write to output file as expected by pipeline: src>>tgt
    in_file = f"{aliases[0]}.gen"
    open(os.path.join('/Users/laabidn1/MolecularTransformer/data/50k_separated/', in_file), 'w').writelines([x+'\n' for rxns in reactions for x in rxns[1]])
    
    # 1. tokenize file: splits the input data (molecules in reactants/products) to predefined tokens
    model = "MIT_mixed_augm_model_average_20.pt"
    dataset = "50k_separated"
    subprocess.run(["/Users/laabidn1/miniconda3/envs/mol_transformer/bin/python", "tokenize_data.py", "-in_file", in_file, "-dataset", dataset])
    
    # # 2. translate: move from reactant to product "sentences"
    beam_size = str(opt.beam_size)
    n_best = str(opt.beam_size)
    translate_output_file = f"experiments/results/predictions_{model}_on_{dataset}_{in_file}.txt"
    src_file = f"data/{dataset}/src-{in_file}.txt"
    model_file = f"experiments/models/{model}"
    subprocess.run(["/Users/laabidn1/miniconda3/envs/mol_transformer/bin/python", "translate.py", "-model", model_file, "-src", src_file, "-output", 
                    translate_output_file, "-batch_size", "4", "-replace_unk", "-max_length", "200", "-beam_size", beam_size, "-n_best", n_best])
    
    # get scores
    targets = [''.join(line.strip().split(' ')) for line in open(f"data/{dataset}/tgt-{in_file}.txt", 'r').readlines()]
    reactants = [''.join(line.strip().split(' ')) for line in open(f"data/{dataset}/src-{in_file}.txt", 'r').readlines()]
    predictions = [[] for i in range(opt.beam_size)]

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    test_df['reactants'] = reactants
    ground_truth = [original_rxn.split('>>')[0] for original_rxn, gen_rxns in reactions for i in range(len(gen_rxns))]
    test_df['ground_truth'] = ground_truth

    # NOTE: the core of this script, in case want to simplify it later
    with open(translate_output_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % opt.beam_size].append(''.join(line.strip().split(' ')))

    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds

    # NOTE: could keep rank just for info; in practice we only care about the true tgt being in any beam
    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', opt.beam_size), axis=1)
    test_df['score'] = pd.concat([test_df['rank']>0, (test_df['reactants']==test_df['ground_truth'])],axis=1).max(axis=1).apply(int)
    
    avg = pd.DataFrame(test_df['target'].unique())
    avg.columns = ['target']
    
    for k in [1, 3, 5, 10]:
        avg[f'round-trip-{k}'] = int(test_df.groupby('target').head(k)['score'].sum()>1)
    
    res = avg.mean(0).to_dict()

    return res
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    parser.add_argument('-wandb_eval_run_id', type=str, required=True,
                       help='The id of the evaluation run on wandb')
    parser.add_argument('-n_conditions', type=str, required=True,
                       help='to identify the samples artifact')
    parser.add_argument('-epoch', type=str, required=True,
                       help='epoch')
    parser.add_argument('-beam_size', type=int, required=True,
                       help='beam_size')
    
    # any other args for identifying the file
    opt = parser.parse_args()
    main(opt)
    