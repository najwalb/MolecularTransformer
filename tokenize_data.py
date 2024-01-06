#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Tokenize data
"""
import argparse
import onmt.opts
import os
import pathlib

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[0]

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def main(opt):
    in_path = os.path.join(parent_path, 'data', opt.dataset, opt.in_file)
    src_path = os.path.join(parent_path, 'data', opt.dataset, 'src-'+opt.in_file+'.txt')
    tgt_path = os.path.join(parent_path, 'data', opt.dataset, 'tgt-'+opt.in_file+'.txt')

    src_tokens = []
    tgt_tokens = []
    for l in open(in_path, 'r').readlines():
        src = l.strip().split('>>')[0]
        tgt = l.strip().split('>>')[1]

        src_tokens.append(smi_tokenizer(src)+'\n')
        tgt_tokens.append(smi_tokenizer(tgt)+'\n')

    open(src_path, 'w').writelines(src_tokens)
    open(tgt_path, 'w').writelines(tgt_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    parser.add_argument('-dataset', type=str, required=True,
                       help='Dataset')
    parser.add_argument('-in_file', type=str, required=True,
                       help='Text file of reaction smiles to tokenize.')
    opt = parser.parse_args()
    main(opt)
    