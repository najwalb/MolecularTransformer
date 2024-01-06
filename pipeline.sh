# Run full pipeline: from 2 text files of reactant (src) and product (tgt) smiles to a prediction score. 
# 3 steps: tokenize, translate and score_predictions.

model=MIT_mixed_augm_model_average_20.pt
dataset=50k_separated
in_file=samples_new.gen
beam_size=100
n_best=100
echo '\nTokenizing...\n'
python tokenize_data.py -in_file ${in_file}\
                        -dataset ${dataset}
echo '\nTranslating...\n'
python translate.py -model experiments/models/${model} \
                    -src data/${dataset}/src-${in_file}.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_${in_file}.txt \
                    -batch_size 2 -replace_unk -max_length 200 \
                    -beam_size ${beam_size} -n_best ${n_best}
                    
echo '\nComputing scores...\n'
python score_predictions.py -beam_size ${beam_size} -targets data/${dataset}/tgt-${in_file}.txt -predictions experiments/results/predictions_${model}_on_${dataset}_${in_file}.txt
