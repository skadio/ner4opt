# Train and Test Data for NER4OPT

Please find the original data repository [here](https://github.com/nl4opt/nl4opt-subtask1-baseline/tree/main/data)

# Data Characteristics

1. The data spans across many domains. 
2. The dataset is curated based on two main categories 
   1. __Source__:advertising, investment, sales domains 
   2. __Target__:production, science, transportation 
3. The training set has __713__ examples solely sampled from the Source domain. 
4. The dev has __99__ examples sampled from both Source and Target domain

## Directory structure
```
├── train.txt  <- Data containing 713 examples tagged in IOB format
├── dev.txt    <- Original dev data containing 99 examples tagged in IOB format used only for testing purpose.
```
