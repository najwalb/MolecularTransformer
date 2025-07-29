#!/bin/bash
#SBATCH --job-name=pipeline_slurm_50k_true.job
#SBATCH --account=project_2006950
#SBATCH --partition=gpu
#SBATCH --output=output/pipeline_slurm_50k_true.out
#SBATCH --error=output/pipeline_slurm_50k_true.err
#SBATCH --time=3-00:00:00
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
HYDRA_FULL_ERROR=1

export WANDB_CACHE_DIR=/scratch/project_2006950/wandb
export WANDB_DATA_DIR=/scratch/project_2006950/wandb
export MPLCONFIGDIR=/scratch/project_2006950
module purge
module load pytorch/2.1
export PYTHONUSERBASE=/projappl/project_2006950/moleculartransformer

# Run full pipeline: from 2 text files of reactant (src) and product (tgt) smiles to a prediction score. 
# 3 steps: tokenize, translate and score_predictions.

model=MIT_mixed_augm_model_average_20.pt
dataset=50k
in_file=test.true
beam_size=100
n_best=100
echo '\nTokenizing...\n'
python tokenize_data.py -in_file ${in_file}\
                        -dataset ${dataset}
# batch size was 2 below
echo '\nTranslating...\n'
python translate.py -model experiments/models/${model} \
                    -src data/${dataset}/src-${in_file}.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_${in_file}.txt \
                    -batch_size 1024 -replace_unk -max_length 200 \ 
                    -beam_size ${beam_size} -n_best ${n_best} \
                    -gpu 0
                    
echo '\nComputing scores...\n'
python score_predictions.py -beam_size ${beam_size} -targets data/${dataset}/tgt-${in_file}.txt -predictions experiments/results/predictions_${model}_on_${dataset}_${in_file}.txt
