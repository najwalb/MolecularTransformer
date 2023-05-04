# Run full pipeline: from 2 text files of reactant (src) and product (tgt) smiles to a prediction score. 
# 3 steps: tokenize, translate and score_predictions.

model=MIT_mixed_augm_model_average_20.pt
dataset=50k_separated
in_file=gen_10
echo '\nTokenizing...\n'
python tokenize_data.py -in_file ${in_file}.csv\
                        -dataset ${dataset}
echo '\nTranslating...\n'
python translate.py -model experiments/models/${model} \
                    -src data/${dataset}/src-${in_file}.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_${in_file}.txt \
                    -batch_size 2 -replace_unk -max_length 200 \
                    -beam_size 5 -n_best 5
echo '\nComputing scores...\n'
python score_predictions.py -targets data/${dataset}/tgt-${in_file}.txt -predictions experiments/results/predictions_${model}_on_${dataset}_${in_file}.txt