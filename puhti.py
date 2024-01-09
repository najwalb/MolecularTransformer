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
run_id = 'l86a0avb'
epochs = '320 260' # [320, 260, 140, 20]
n_conditions = 128
n_samples_per_condition = 100
edge_conditional_set = 'val'
beam_size = 100
n_best = 100
batch_size = 256

experiment_name = f'wandb_pipeline_{run_id}'
print(f"Creating job {experiment_name}... ")
job_file = os.path.join(job_directory, f"{experiment_name}.job")

# TODO: Could load the yaml file in question the experiment name and log with that locally to outputs/
with open(job_file, 'w') as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines(f"#SBATCH --job-name={experiment_name}_%a.job\n") # add time stamp?
    fh.writelines(f"#SBATCH --output={output_dir}/{experiment_name}_%a.out\n")
    fh.writelines(f"#SBATCH --error={output_dir}/{experiment_name}_%a.err\n")
    fh.writelines(f"#SBATCH --account={project}\n")
    fh.writelines(f"#SBATCH --partition=gpu\n")
    fh.writelines(f"#SBATCH --gres=gpu:v100:{num_gpus}\n")
    fh.writelines("#SBATCH --mem-per-cpu=10G\n")
    fh.writelines("#SBATCH --cpus-per-task=16\n")
    fh.writelines(f"#SBATCH --time=06:00:00\n")
    fh.writelines("#SBATCH --array=1-1\n")
    fh.writelines("module purge\n")
    fh.writelines("module load gcc/11.3.0\n\n")
    fh.writelines(f"export WANDB_CACHE_DIR=/scratch/{project}\n")
    fh.writelines(f"export MPLCONFIGDIR=/scratch/{project}\n")
    fh.writelines(f'export PATH="/projappl/{project}/{conda_env}/bin:$PATH"\n')
    fh.writelines(f"python3 {script_name} -wandb_run_id {run_id} -n_conditions {n_conditions} -beam_size {beam_size} -n_best {n_best} "+\
                    f" -batch_size {batch_size} -epochs {epochs} -edge_conditional_set {edge_conditional_set} -n_samples_per_condition {n_samples_per_condition} ")

result = subprocess.run(args="sbatch", stdin=open(job_file, 'r'), capture_output=True)
if 'job' not in result.stdout.decode("utf-8"):
    print(result)
else:
    job_id = result.stdout.decode("utf-8").strip().split('job ')[1]

    with open(jobids_file, 'a') as f:
        f.write(f"train.job: {job_id}\n")

    print(f"=== Submitted to Slurm with ID {job_id}.")
