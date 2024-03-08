import os
import subprocess

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.makedirs(dir)
        
conda_env = "moltransformer"
script_name = "wandb-pipeline.py" # train.py
project = "project_2006950"
job_directory = os.path.join('outputs/', script_name.split('.py')[0])

output_dir = os.path.join(job_directory, 'out')
jobids_file = os.path.join(job_directory, 'jobids.txt')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(output_dir)

num_gpus = 1
# run_id = 'l86a0avb'
# epochs = '320 260 140 20' # [320, 260, 140, 20]
# run_id = 's82o6pi4'
# epochs = '150 100 20'
# run_id = 'if3aizpe'
# epochs = '200 240'
# run_id = 'o6zn38fc' # charged_smiles_pos_enc
# epochs = '440 480'
# run_id = 'z7d425l4' # p_to_r_init_10
# epochs = '440 420'
# run_id = 'z7d425l4' # p_to_r_init_10
# epochs = '320'
# n_conditions = 5000 # 

##### skip connection model
# run_id = '6q1cd3e3' # uncharged_smiles_pos_enc
# epochs = '260'
# n_conditions = 4960 # 
# n_samples_per_condition = 100
# steps = 250
# edge_conditional_set = 'test'
# beam_size = 100
# n_best = 100
# batch_size = 64

##### best laplacian model
# run_id = 'if3aizpe' # uncharged_smiles_pos_enc
# epochs = '300'
# n_conditions = 4960 # 
# n_samples_per_condition = 100
# steps = 100
# edge_conditional_set = 'test'
# beam_size = 100
# n_best = 100
# batch_size = 64

####### TODO: new MT hyperparams from retrobridge

##### skip connection model
# run_id = '6q1cd3e3' # uncharged_smiles_pos_enc
# epochs = '260'
# n_conditions = 4960 # 
# n_samples_per_condition = 100
# steps = 250
# edge_conditional_set = 'test'
# beam_size = 100
# n_best = 100
# batch_size = 64

##### laplacian model
# run_id = 'if3aizpe' # uncharged_smiles_pos_enc
# epochs = '300'
# n_conditions = 4960 # 
# n_samples_per_condition = 100
# steps = 100
# edge_conditional_set = 'test'
# beam_size = 100
# n_best = 100
# batch_size = 64

##### 250 steps model
# run_id = 'bznufq64' # uncharged_smiles_pos_enc
# epochs = '200' 
# n_conditions = 4992 # 
# n_samples_per_condition = 100
# steps = 250
# edge_conditional_set = 'test' #
# reprocess_like_retrobridge = True
# keep_unprocessed = True
# remove_charges = True
# beam_size = 100
# n_best = 100
# batch_size = 64

##### new runs

all_run_ids = ['7ckmnkvc', '82wscv5d', 'bznufq64', 'p28j1qh2', 'sne65sw8', '7ckmnkvc', '82wscv5d', '9nd48syv', 'bznufq64', 'k6di2qtr',
           'p28j1qh2', 'sne65sw8']
all_epochs = ['360', '360', '360', '360', '360', '280', '280', '280', '280', '280', '280', '280']


all_run_ids = ['7ckmnkvc']
all_epochs = ['360']

all_n_conditions = [4992]*len(all_run_ids)
all_n_samples_per_condition = [100]*len(all_run_ids)
all_steps = [100]*len(all_run_ids)
all_edge_conditional_set = ['val']*len(all_run_ids)
all_new_prob_weights = [0.9]*len(all_run_ids)
all_ranking_metrics = ['new_weighted_prob']*len(all_run_ids)
all_boolean_flags = ['--reprocess_like_retrobridge --keep_unprocessed --remove_charges --log_to_wandb']*len(all_run_ids)
all_boolean_flags = ['--reprocess_like_retrobridge --keep_unprocessed --log_to_wandb']*len(all_run_ids)

for (run_id, epochs, n_conditions, n_samples_per_condition, steps, edge_conditional_set, new_prob_weight, ranking_metric, boolean_flags) \
    in zip(all_run_ids, all_epochs, all_n_conditions, all_n_samples_per_condition, all_steps, all_edge_conditional_set, \
           all_new_prob_weights, all_ranking_metrics, all_boolean_flags):

    reprocess_like_retrobridge = 'reprocess_like_retrobridge' in boolean_flags
    keep_unprocessed = 'keep_unprocessed' in boolean_flags
    remove_charges = 'remove_charges' in boolean_flags
    experiment_name = f'{run_id}_wandb_pipeline_retrobridge_reproc{reprocess_like_retrobridge}_keep{keep_unprocessed}_rmch{remove_charges}_nprob{new_prob_weight}_rank{ranking_metric}'
    print(f"Creating job {experiment_name}... ")
    job_file = os.path.join(job_directory, f"{experiment_name}.job")

    # TODO: Could load the yaml file in question the experiment name and log with that locally to outputs/
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={experiment_name}_%a.job\n") # add time stamp?
        fh.writelines(f"#SBATCH --output={output_dir}/{experiment_name}_%a.out\n")
        fh.writelines(f"#SBATCH --error={output_dir}/{experiment_name}_%a.err\n")
        fh.writelines(f"#SBATCH --account={project}\n")
        # fh.writelines(f"#SBATCH --partition=gpu\n")
        # fh.writelines(f"#SBATCH --gres=gpu:v100:{num_gpus}\n")
        fh.writelines("#SBATCH --mem-per-cpu=10G\n")
        fh.writelines("#SBATCH --cpus-per-task=2\n")
        fh.writelines(f"#SBATCH --time=10:00:00\n")
        fh.writelines("#SBATCH --array=1-1\n")
        fh.writelines("module purge\n")
        fh.writelines("module load gcc/11.3.0\n\n")
        fh.writelines(f"export WANDB_CACHE_DIR=/scratch/{project}\n")
        fh.writelines(f"export MPLCONFIGDIR=/scratch/{project}\n")
        fh.writelines(f'export PATH="/projappl/{project}/{conda_env}/bin:$PATH"\n')
        fh.writelines(f"python3 retrobridge_like_roundtrip.py --wandb_run_id {run_id} --n_conditions {n_conditions} --steps {steps} --epochs {epochs} --edge_conditional_set {edge_conditional_set} --n_samples_per_condition {n_samples_per_condition}"+\
                      f" --new_prob_weight {new_prob_weight} --ranking_metric {ranking_metric} {boolean_flags}\n")

    # result = subprocess.run(args="sbatch", stdin=open(job_file, 'r'), capture_output=True)
    # if 'job' not in result.stdout.decode("utf-8"):
    #     print(result)
    # else:
    #     job_id = result.stdout.decode("utf-8").strip().split('job ')[1]

    #     with open(jobids_file, 'a') as f:
    #         f.write(f"train.job: {job_id}\n")

    #     print(f"=== Submitted to Slurm with ID {job_id}.")
