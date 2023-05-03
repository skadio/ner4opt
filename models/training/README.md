# This folder has all our artifacts related to training

- Data for training and test
- Our best hybrid model training script
- Log of our result on our infrastructure
- Other related notebooks

## Training script for our best model

```
# Install dependencies for calculating F-1: Original NL4OPT Utils
pip install allennlp overrides

# Install Ner4Opt
pip install ner4opt

# Run training and evaluation
python train_and_evaluate_hybrid_model.py --train "data/train.txt" --test "data/dev.txt"
```

## Directory structure
```
├── data                                <- Data for training and testing
├── logs                                <- Our result logs
    ├── hybrid_log.txt                  <- Log showing our best metrics using the hybrid model
├── nl4opt_utils                        <- Utils from NL4Opt. Please see below for the original folder
├── notebooks                           <- Notebooks for data augmentation and gazetteers
├── train_and_evaluate_hybrid_model.py  <- Hybrid model training and evaluation script
```

[Original Utils from NL4OPT](https://github.com/nl4opt/nl4opt-subtask1-baseline/tree/main/baseline/utils)

_Please note, there would be some level of randomness in the results due to the nature of the transformers model and randomsearchcv_
