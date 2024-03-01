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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[0]


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
    parser.add_argument("--translation_out_file", type=Path, required=False, default=True)
    parser.add_argument("--topk", type=list, default=[1, 3, 5, 10, 50])
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
    args = parser.parse_args()
    args = parser.parse_args()

    translation_out_file = os.path.join('experiments', 'results', 'retroB_translated_eval_epoch200_steps250_resorted_0.9_cond4992_sampercond100_test_lam0.9_retrobridge_roundtrip_parsed.txt')
    translation_out_file = '/scratch/project_2006950/MolecularTransformer/data/retrobridge/restrobridge_samples.csv'
    df = pd.read_csv(translation_out_file)
    
    df['exact_match'] = df['true'] == df['pred']
    df['round_trip_match'] = df['product'] == df['pred_product']
    df['match'] = df['exact_match'] | df['round_trip_match']
    
    avg = {}
    for k in args.topk:
        avg[f'roundtrip-coverage-{k}_weighted_0.9'] = df.groupby('product').head(int(k)).groupby('product').match.any().reset_index()['match']
        avg[f'roundtrip-accuracy-{k}_weighted_0.9'] = df.groupby('product').head(int(k)).groupby('product').match.mean().reset_index()['match']
        
    avg = {k:v.mean(0) for k,v in avg.items()}
    avg['epochs'] = args.epochs
    
    # 8. log scores to wandb
    wandb_entity = 'najwalb'
    wandb_project = 'retrodiffuser'
    model = "MIT_mixed_augm_model_average_20.pt"
    # run = wandb.init(name=f"round_trip_{args.wandb_run_id}_cond{args.n_conditions}_sampercond{args.n_samples_per_condition}_{args.edge_conditional_set}_retrobridgeRoundTrip", 
    #                  project=wandb_project, entity=wandb_entity, resume='allow', job_type='round_trip', config={"experiment_group": args.wandb_run_id})
    
    run = wandb.init(name=f"retrobridge_samples", 
                     project=wandb_project, entity=wandb_entity, resume='allow', job_type='round_trip', config={"experiment_group": args.wandb_run_id})
    # log.info('Logging to wandb...')
    #t0 = time.time()
    run.log({'round_trip_k/': avg})
    savedir = os.path.join(parent_path, "data", f"{args.wandb_run_id}_eval")
    eval_file, artifact_name = donwload_eval_file_from_artifact(wandb_entity, wandb_project, args.wandb_run_id, args.epochs[0], args.steps, 
                                                                args.n_conditions, args.n_samples_per_condition, args.edge_conditional_set, savedir)
    run.use_artifact(artifact_name)
    # artifact = wandb.Artifact(f'{args.wandb_run_id}_round_trip', type='round_trip')
    # eval_file_name = eval_file.split('/')[-1].split('.txt')[0]
    # art_name = f"round_trip_{eval_file_name}_10"
    # artifact.add_file(round_trip_output_path, name=f'{art_name}.txt')
    # run.log_artifact(artifact, aliases=[art_name])
    
    # log.info(f'Time of wandb logging {time.time()-t0}\n')



    
    
