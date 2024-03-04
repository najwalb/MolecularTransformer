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

experiment_name = f'retrobridge_retranslating'
print(f"Creating job {experiment_name}... ")
job_file = os.path.join(job_directory, f"{experiment_name}.job")

csv_in = "/scratch/project_2006950/MolecularTransformer/data/retrobridge/retrobridge_samples_without_translation.csv"
csv_out = "/scratch/project_2006950/MolecularTransformer/data/retrobridge/retrobridge_samples_retranslated.csv"

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
    fh.writelines(f"#SBATCH --time=1-00:00:00\n")
    fh.writelines("#SBATCH --array=1-1\n")
    fh.writelines("module purge\n")
    fh.writelines("module load gcc/11.3.0\n\n")
    fh.writelines(f"export WANDB_CACHE_DIR=/scratch/{project}\n")
    fh.writelines(f"export MPLCONFIGDIR=/scratch/{project}\n")
    fh.writelines(f'export PATH="/projappl/{project}/{conda_env}/bin:$PATH"\n')
    fh.writelines(f"python3 retrobridge_code_roundtrip.py --csv_file {csv_in} --csv_out {csv_out}")

result = subprocess.run(args="sbatch", stdin=open(job_file, 'r'), capture_output=True)
if 'job' not in result.stdout.decode("utf-8"):
    print(result)
else:
    job_id = result.stdout.decode("utf-8").strip().split('job ')[1]

    with open(jobids_file, 'a') as f:
        f.write(f"train.job: {job_id}\n")

    print(f"=== Submitted to Slurm with ID {job_id}.")
