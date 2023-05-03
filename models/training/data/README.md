# Train and Test Data for NER4OPT

Please download the data required (train & dev) from the original repository - [NL4OPT @ NeurIPS'22 Data Repository](https://github.com/nl4opt/nl4opt-subtask1-baseline/tree/main/data)

# Data Characteristics

1. The data spans across many domains
2. The dataset is curated based on two main categories
   1. __Source__: advertising, investment, sales domains
   2. __Target__: production, science, transportation
3. The training set has __713__ examples solely sampled from the Source domain
4. The dev has __99__ examples sampled from both Source and Target domain
5. All the examples in the data (train and dev) are **spacy tokenized**, so one would observe certain tokens like $, % etc., split from the preceding tokens.

# To convert the data to json, you can use the following
```
python -m spacy convert train.txt . -t json -n 1 -c iob
python -m spacy convert dev.txt . -t json -n 1 -c iob
```
The reason for this conversion is to easily identify frequently occurring key-phrases to engineer relevant features.
For more details, please refer our [jupyter notebook](https://github.com/skadio/ner4opt/models/training/notebooks/entity_gazetteers.ipynb)
