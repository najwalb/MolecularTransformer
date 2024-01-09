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
import pickle
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
    
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

def get_artifact_with_newest_alias(wandb_entity, wandb_project, wandb_run_id, collection_name, epoch, n_conditions, n_samples_per_condition, edge_conditional_set):
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
                     if ('eval' in alias and 'cond' in alias) \
                     and re.findall('epoch\d+', alias)[0]==f'epoch{epoch}' \
                     and re.findall('cond\d+', alias)[0]==f'cond{n_conditions}' \
                     and re.findall('sampercond\d+', alias)[0]==f'sampercond{n_samples_per_condition}' \
                     and alias.split('_')[6].split('.txt')[-1]==edge_conditional_set]
    versions = [int(art.version.split('v')[-1]) for art in coll.versions()]

    aliases = [a for a,v in sorted(zip(aliases, versions), key=lambda pair: -pair[1])]
    
    assert len(aliases)>0, f'No alias found.'
   
    return aliases[0]
    
def donwload_eval_file_from_artifact(wandb_entity, wandb_project, wandb_run_id, epoch, n_conditions, n_samples_per_condition, edge_conditional_set, savedir):
    collection_name = f"{wandb_run_id}_eval"
    alias = get_artifact_with_newest_alias(wandb_entity, wandb_project, wandb_run_id, collection_name, 
                                           epoch, n_conditions, n_samples_per_condition, edge_conditional_set)
    
    artifact_name = f"{wandb_entity}/{wandb_project}/{collection_name}:{alias}"
    samples_art = wandb.Api().artifact(artifact_name)
    samples_art.download(root=savedir)
    
    return os.path.join(savedir, f"{alias}.txt"), artifact_name
    
def output_round_trip_scores(df, output_file):
    '''
        Output file of the format:
            (cond #) original_rxn:
                gen_rxn, exact_match_score (0/1), round_trip score (0/1), 1st MT prediction, 2nd MT pred, 3rd MT pred
    '''
    targets = df['target'].unique()
    ground_truth = df['ground_truth'].unique()
 
    f = open(output_file, 'w')
    for i, (p, g) in enumerate(zip(targets, ground_truth)):
        f.write(f'(cond {i}) {g}>>{p}:\n')
        df_p = df[df['target']==p].reset_index()
        for j in range(len(df_p)):
            f.write(f"\t{df_p.loc[j,'reactants']}>>{p}, match: {df_p.loc[j,'exact_match_score']}, "+\
                    f"round-trip: {df_p.loc[j,'round_trip_score']}, pred1: {df_p.loc[j,'prediction_1']}, "+\
                    f"pred2: {df_p.loc[j,'prediction_2']}, pred3: {df_p.loc[j,'prediction_3']}\n")
    f.close()
    
def main(opt):
    ''' 
        Logic:
            => 1 point if: exact matches, round trip matches 
            => round trip top k: if top k has a score 1
            => keep in same order as given in elbo
    '''
    wandb_entity = 'najwalb'
    wandb_project = 'retrodiffuser'
    model = "MIT_mixed_augm_model_average_20.pt"
    
    # conda_env_path = '/Users/laabidn1/miniconda3/envs/mol_transformer/bin/' # used by subprocess
    # TODO: add sample_step to the run name once it's in wandb artifacts
    run = wandb.init(name=f"round_trip_{opt.wandb_run_id}_cond{opt.n_conditions}_sampercond{opt.n_samples_per_condition}_{opt.edge_conditional_set}", 
                     project=wandb_project, entity=wandb_entity, resume='allow', job_type='round_trip', config={"experiment_group": opt.wandb_run_id})
    
    # 1. get the evaluation file from wandb
    for epoch in opt.epochs:
        log.info(f'Getting eval file from wandb...\n')
        savedir = os.path.join(parent_path, "data", f"{opt.wandb_run_id}_eval")
        log.info(f'==== Saving artifact in {savedir}\n')
        eval_file, artifact_name = donwload_eval_file_from_artifact(wandb_entity, wandb_project, opt.wandb_run_id, epoch, opt.n_conditions, 
                                                                    opt.n_samples_per_condition, opt.edge_conditional_set, savedir)
    
        # 2. read saved reaction data from eval file
        log.info(f'Read saved reactions...\n')
        reactions = read_saved_reaction_data(eval_file)
    
        # 3. output reaction data to text file as one reaction per line
        log.info(f'Output reactions to file for processing...\n')
        eval_file_name = eval_file.split('/')[-1].split('.txt')[0]
        dataset = f'{opt.wandb_run_id}_eval' # use eval_run name (collection name) as dataset name
        reaction_file_name = f"{eval_file_name}_TOKENIZED.gen"
        reaction_file_path = os.path.join(parent_path, 'data', dataset, reaction_file_name) 
        log.info(f'==== reaction_file_path {reaction_file_path}\n')
        open(reaction_file_path, 'w').writelines([x+'\n' for rxns in reactions for x in rxns[1]])
    
        # 4. tokenize file: splits the input data (molecules in reactants/products) to predefined tokens
        log.info(f'Tokenizing ...\n')
        t0 = time.time()
        subprocess.run(["python", "tokenize_data.py", "-in_file", reaction_file_name, "-dataset", dataset], stdout=None, stderr=None)
        log.info(f'Time tokenizing {time.time()-t0}\n')
    
        # 5. translate: move from reactant to product "sentences"
        log.info(f'Translating ...\n')
        t0 = time.time()
        translate_output_file = f"experiments/results/predictions_{model}_on_{dataset}_{reaction_file_name}.txt"
        src_file = f"data/{dataset}/src-{reaction_file_name}.txt"
        model_file = f"experiments/models/{model}"
        subprocess.run(["python", "translate.py", "-model", model_file, "-src", src_file, "-output", 
                        translate_output_file, "-batch_size", opt.batch_size, "-replace_unk", "-max_length", 
                        opt.max_length, "-beam_size", opt.beam_size, "-n_best", opt.n_best], stdout=None, stderr=None)
        log.info(f'Time translating {time.time()-t0}\n')
    
        # 6. get round_trip k accuracy scores
        log.info(f'Computing top-k ...\n')
        targets = [''.join(line.strip().split(' ')) for line in open(f"data/{dataset}/tgt-{reaction_file_name}.txt", 'r').readlines()]
        reactants = [''.join(line.strip().split(' ')) for line in open(f"data/{dataset}/src-{reaction_file_name}.txt", 'r').readlines()]
        predictions = [[] for i in range(int(opt.beam_size))]

        test_df = pd.DataFrame(targets)
        test_df.columns = ['target']
        test_df['reactants'] = reactants
        ground_truth = [original_rxn.split('>>')[0] for original_rxn, gen_rxns in reactions for i in range(len(gen_rxns))]
        test_df['ground_truth'] = ground_truth

        # NOTE: the core of this script, in case want to simplify it later
        log.info(f'Reading out predictions using beam search ...\n')
        t0 = time.time()
        with open(translate_output_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                predictions[i % int(opt.beam_size)].append(''.join(line.strip().split(' ')))

        for i, preds in enumerate(predictions):
            test_df['prediction_{}'.format(i + 1)] = preds
        log.info(f'Time reading predictions {t0 - time.time()}\n')

        log.info(f'Calculating top-k ...\n')
        t0 = time.time()
        # NOTE: could keep rank just for info; in practice we only care about the true tgt being in any beam
        test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', int(opt.beam_size)), axis=1)
        # score = 1 if round_trip matches (i.e. rank>0) or ground_truth matches (exact equality)
        test_df['exact_match_score'] = (test_df['reactants']==test_df['ground_truth']).apply(int)
        test_df['round_trip_score'] = (test_df['rank']>0).apply(int)
        test_df['score'] = pd.concat([test_df['round_trip_score'], test_df['exact_match_score']], axis=1).max(axis=1)
        
        log.info(f'Output top-k files ...\n')
        round_trip_output_name = f"round_trip_{eval_file_name}"
        round_trip_output_path = os.path.join(parent_path, "data", dataset, f"{round_trip_output_name}.txt")
        output_round_trip_scores(test_df, round_trip_output_path)
    
        # 7. compute top-k averages
        avg = pd.DataFrame(test_df['target'].unique())
        avg.columns = ['target']
        
        for k in opt.round_trip_k:
            avg[f'round-trip-{k}_weighted_0.9'] = (test_df.groupby('target').head(int(k)).groupby('target').agg({'score':'sum'})>=1).reset_index()['score']
        
        round_trip_res = avg.mean(0).to_dict()
        round_trip_res['epoch'] = epoch
        log.info(f'round_trip_res {round_trip_res}\n')
        log.info(f'Time computing topk {time.time()-t0}\n')
        
        # 8. log scores to wandb
        log.info('Logging to wandb...')
        t0 = time.time()
        run.log({'round_trip_k/': round_trip_res})
        run.use_artifact(artifact_name)
        artifact = wandb.Artifact(f'{opt.wandb_run_id}_round_trip', type='round_trip')
        artifact.add_file(round_trip_output_path, name=f'{round_trip_output_name}.txt')
        run.log_artifact(artifact, aliases=[round_trip_output_name])
        
        log.info(f'Time of wandb logging {time.time()-t0}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    parser.add_argument('-wandb_run_id', type=str, required=True,
                       help='The id of the training run on wandb')
    parser.add_argument('-n_conditions', type=str, required=True,
                       help='to identify the samples artifact')
    parser.add_argument('-epochs', nargs='+', type=int, required=True,
                       help='epochs')
    parser.add_argument('-round_trip_k', nargs='+', type=int, default=[1, 3, 5, 10, 50],
                       help='list of k values for the round_trip accuracy')
    parser.add_argument('-n_samples_per_condition', type=str, default="100",
                       help='nb samples per condition')
    parser.add_argument('-edge_conditional_set', type=str, default='test',
                       help='edge conditional set')
    parser.add_argument('-beam_size', type=str, default="10",
                       help='beam_size')
    parser.add_argument('-n_best', type=str, default="10",
                       help='n_best')
    parser.add_argument('-max_length', type=str, default="200",
                       help='max_length')
    parser.add_argument('-batch_size', type=str, default="2",
                       help='batch_size')
    
    # any other args for identifying the file
    opt = parser.parse_args()
    main(opt)
    