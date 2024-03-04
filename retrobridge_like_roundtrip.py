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

import re

def remove_charges_func(reactant):
    # Regular expression to match charges (+, -, and numerical charges like +2, -3, etc.)
    charge_pattern = re.compile(r'(\d+)?[\+\-]')
    # Remove charges from the reactant string
    cleaned_reactant = re.sub(charge_pattern, '', reactant)
    # print(reactant, cleaned_reactant)
    return cleaned_reactant

"""
Chen, Shuan, and Yousung Jung.
"Deep retrosynthetic reaction prediction using local reactivity and global attention."
JACS Au 1.10 (2021): 1612-1620.

Predicted precursors are considered correct if 
- the predicted precursors are the same as the ground truth
- Molecular Transformer predicts the target product for the proposed precursors
"""
'''
python3 retrobridge-roundtrip.py --csv_out testing_wandb.out --wandb_run_id bznufq64 --n_conditions 4992 --steps 250 --epochs 200 --edge_conditional_set test --n_samples_per_condition 100
'''
def read_saved_reaction_data_like_retrobridge(output_file):
    # Reads the saved reaction data from the samples.txt file
    # Split the data into individual blocks based on '(cond ?)' pattern
    data = open(output_file, 'r').read()
    blocks = re.split(r'\(cond \d+\)', data)[1:]
    reactant_predictions = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        original_reaction = lines[0].split(':')[0].strip()
        prod = remove_charges_func(original_reaction.split('>>')[-1])
        true_reactants = remove_charges_func(original_reaction.split('>>')[0])
        for line in lines[1:]:
            match = re.match(r"\t\('([^']+)', \[([^\]]+)\]\)", line)
            if match:
                reaction_smiles = match.group(1)
                pred_reactants = remove_charges_func(reaction_smiles.split('>>')[0])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--csv_file", type=Path, required=True)
    parser.add_argument("--csv_out", type=Path, required=False, default=True)
    parser.add_argument("--mol_trans_dir", type=Path, default="./")
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
    eval_file, artifact_name = donwload_eval_file_from_artifact(wandb_entity, wandb_project, args.wandb_run_id, epoch, args.steps, 
                                                                args.n_conditions, args.n_samples_per_condition, args.edge_conditional_set, 
                                                                savedir)
    
    # 2. read saved reaction data from eval file
    log.info(f'Read saved reactions...\n')
    reactions = read_saved_reaction_data_like_retrobridge(eval_file)

    # 3. output reaction data to text file as one reaction per line
    mt_opts = f'retrobridge_roundtrip'
    log.info(f'Output reactions to file for processing...\n')
    eval_file_name = eval_file.split('/')[-1].split('.txt')[0]
    dataset = f'{args.wandb_run_id}_eval' # use eval_run name (collection name) as dataset name
    reaction_file_name = f"{eval_file_name}_{mt_opts}_parsed.txt"
    reaction_file_path = os.path.join(parent_path, 'data', dataset, reaction_file_name) 
    log.info(f'==== reaction_file_path {reaction_file_path}\n')
    open(reaction_file_path, 'w').write('product,true,pred\n')
    open(reaction_file_path, 'a').writelines([f'{prod},{true},{pred}\n' for prod,true,pred in reactions])
    
    # Read CSV
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
    pred_products = {r: p for r, p in zip(unique_smiles, pred_products)}

    # update dataframe
    df['pred_product'] = [pred_products[r] for r in df['pred']]

    # Write results
    if args.csv_out:
        print("Writing CSV file...")
        output_file = f'retroB_translated_{reaction_file_name}'
        df.to_csv(os.path.join(f'experiments/results/{output_file}'), index=False)
