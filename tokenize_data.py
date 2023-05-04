#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Tokenize data
"""

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

def main():
    in_path = '/Users/laabidn1/MolecularTransformer/data/50k_separated/gen_10.csv'
    src_path = '/Users/laabidn1/MolecularTransformer/data/50k_separated/src-gen-10.txt'
    tgt_path = '/Users/laabidn1/MolecularTransformer/data/50k_separated/tgt-gen-10.txt'

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
    main()