# complexity-scaling
Does compressibility predict data-sensitive scaling laws?

WORK IN PROGRESS: this repo will be refactored!

Experiments run on an RTX 3080

next training run todo:
- [X] capture loss of every batch instead of perplexity at epoch level
    - [X] reduce epochs for larger datasets?
- [X] script to run each dataset on separate GPU
- [X] learning rate decay
- [X] reduce validation size?
- [X] standardize dataset and upload/download huggingface

are current codegen models under or over parameterized? based on idea that code is more compressible
why do the 2 least compressible datasets end up getting the lowest perplexity when FLOPS are maxxed out (see gif)?

NOTE:
from current runs (3/26 4:20pm), we're going to be missing 0.61-1.4B-50M and 0.52-1.4B-20M due to server kill
Total = 5 * 6 * 3 + 1 * 6 * 5 = 120 runs
- delete 2 duplicate 0.61-4.4M-50M and 0.61-8.8M-50M runs
- missing 0.61-1.4B-50M and 0.52-1.4B-20M (just use the 50% and 20% of the 100M runs, duh)
- adjusted total = 120-2-2+1+2 = 119 runs

TODO:
- [X] plot Chinchilla Approach 1 (performance over train steps)
- [ ] compute and plot optimal frontier (ideally D_opt(N)) for datablations log-space L functional form
    - we have the original chinchilla L functional form and its D_opt(C), and a related derivation in Kaplan appendix
    - [ ] need to find similar closed-form optimal frontier in log-space
- [X] ablation on syntax with vocab size the same
    - [X] also, 4 datasets with roughly similar entropy but slightly different PCFG hparams
- [X] normalize natlang and code samples to max seq len; measure compressibility
- codegen model stuff
    - [X] prep and upload dataset
    - [X] run scaling exps on codegen dataset
    - compute scaling laws
        - [X] empirically from scaling exps
        - [X] from chinchilla optimal
            - 400M params with 8B tokens
            - [data-generating] sample code (C) dataset of this size
            - [ ] train chinchilla-optimal code model
                - will need to set up multi-gpu training for this
        - from gzip optimal
    - natural language chinchilla replication
    - comment removal, variable name removal
- [X] experiment with subsets of diff gzipability of natlang/code datasets
    - algorithm for creating subsets of diff gzipability distributions?
    - [X] HTML=0.35, C=0.37, Python=0.41
- theoretical compressibility / info theory stuff
- get rid of data portion loop since we store loss from each step