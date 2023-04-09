# Training script for our best model

```
pip install ner4opt
chmod +x train_and_evaluate_hybrid_model.sh
./train_and_evaluate_hybrid_model.sh
```

## Directory structure
```
├── logs                                <- Our Result Log
    ├── hybrid_log.txt                  <- Log showing our best metrics using the hybrid model
├── nl4opt_utils                        <- Utils from NL4Opt. Please see below for the original folder
├── train_and_evaluate_hybrid_model.py  <- Hybrid model training and evaluation script
├── train_and_evaluate_hybrid_model.sh  <- Bash Script for running our Hybrid model end-to-end
```

[Original Utils from NL4OPT](https://github.com/nl4opt/nl4opt-subtask1-baseline/tree/main/baseline/utils)

_Please note, there would be some level of randomness due to the nature of the transformers model and randomsearchcv_
