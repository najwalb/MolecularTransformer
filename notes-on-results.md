# RB on common products, with confidence
- run: `python3 compare_rb_to_7ck620.py` #TODO: refactor this
- run: `python3 retrobridge_like_score_roundtrip.py  --log_to_wandb --input_file data/rb_vs_7ck620/samples_from_rb_in_common.csv`
coverage:
	{'top-1': '0.8613273248668578\n', 'top-3': '0.9688652191724703\n', 'top-5': '0.9832036050798852\n', 'top-10': '0.9916018025399427\n', 'top-50': '0.9944694797214256\n'}
accuracie:
	{'top-1': '0.8613273248668578\n', 'top-3': '0.7453229550730575\n', 'top-5': '0.6869520688242523\n', 'top-10': '0.6175130703653849\n', 'top-50': '0.5750132173745199\n'}
## results when using group as the key to compute confidence
coverage:
	{1: '0.8610599549826069\n', 3: '0.9686924493554327\n', 5: '0.9832207898506241\n', 10: '0.991610394925312\n', 50: '0.994475138121547\n'}

accuracy:
	{1: '0.8610599549826069\n', 3: '0.7452424800491099\n', 5: '0.6868221812973194\n', 10: '0.6173722964392087\n', 50: '0.5749526525837354\n'}


# 7ck620 on common products (original), with confidence
- run: `python3 compare_rb_to_7ck620.py` (no need to redo)
- run: `python3 retrobridge_like_score_roundtrip.py --log_to_wandb --input_file data/rb_vs_7ck620/samples_from_7ck620_in_common.csv`
coverage:
	{1: '0.8201556739041377\n', 3: '0.9037279803359279\n', 5: '0.9215485456780008\n', 10: '0.9348627611634576\n', 50: '0.9451044653830397\n'}

accuracy:
	{1: '0.8201556739041377\n', 3: '0.6298989485183668\n', 5: '0.5357503755291547\n', 10: '0.4327927989439666\n', 50: '0.35511171517627715\n'}

# 7ck620 on common products (reconstructed), with confidence
- reconstructed = rebuilding all molecules (prods+rcts) from atoms and bonds only
  - to get a representation matching that of rb
- run: `python3 compare_rb_to_7ck620.py` (no need to redo) ?
- run: `python3 retrobridge_like_score_roundtrip.py --log_to_wandb --input_file data/rb_vs_7ck620/samples_from_7ck620_in_common_all_reconstructed.csv`
coverage:
	{1: '0.8502662843097092\n', 3: '0.9410077836952069\n', 5: '0.9602621876280213\n', 10: '0.9750102417042196\n', 50: '0.9838181073330602\n'}
accuracy:
	{1: '0.8502662843097092\n', 3: '0.6746551959579407\n', 5: '0.5897514679776048\n', 10: '0.49466748600300425\n', 50: '0.42040976446910944\n'}


avg {'roundtrip-coverage-1_weighted_0.9': 0.5, 'roundtrip-accuracy-1_weighted_0.9': 0.5, 'roundtrip-coverage-3_weighted_0.9': 1.0, 'roundtrip-accuracy-3_weighted_0.9': 0.8333333333333333, 'roundtrip-coverage-5_weighted_0.9': 1.0, 'roundtrip-accuracy-5_weighted_0.9': 0.8, 'roundtrip-coverage-10_weighted_0.9': 1.0, 'roundtrip-accuracy-10_weighted_0.9': 0.8799999999999999, 'roundtrip-coverage-50_weighted_0.9': 1.0, 'roundtrip-accuracy-50_weighted_0.9': 0.8003892104515338}