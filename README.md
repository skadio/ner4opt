# Ner4Opt: Named Entity Recognition for Optimization 

Given an optimization problem in natural language, Ner4Opt extracts optimization related entities from free-form text. 

See our [HugginFace Space Demo](https://huggingface.co/spaces/skadio/Ner4Opt) to test it yourself 🤗. 

The Ner4Opt model ([CPAIOR'23](https://github.com/skadio/ner4opt/blob/main/docs/%5BCPAIOR%202023%5D%20Ner4Opt%20Paper.pdf), [NeurIPS'22](https://github.com/skadio/ner4opt/blob/main/docs/%5BNeurIPS%202022%5D%20Ner4Opt%20Poster.pdf)) is trained to detect six named entities: 

- **CONST_DIR**: Constraint direction
- **LIMIT**: Limit
- **OBJ_DIR**: Objective direction
- **OBJ_NAME**: Objective name
- **PARAM**: Parameter
- **VAR**: Variable

Here are the details of our [pretrained models](https://github.com/skadio/ner4opt/blob/main/models/README.md) and the [training procedure](https://github.com/skadio/ner4opt/blob/main/models/training/README.md). Large pretrained models are hosted in [HuggingFace Models](https://huggingface.co/skadio).

## Quick Start

```python
# Import
from ner4opt import Ner4Opt

# Input optimization problem description as free-form text
problem_description = "Cautious Asset Investment has a total of $150,000 to manage and decides to invest it in money market fund, which yields a 2% return as well as in foreign bonds, which gives and average rate of return of 10.2%. Internal policies require PAI to diversify the asset allocation so that the minimum investment in money market fund is 40% of the total investment. Due to the risk of default of foreign countries, no more than 40% of the total investment should be allocated to foreign bonds. How much should the Cautious Asset Investment allocate in each asset so as to maximize its average return?"

# Ner4Opt Model with options lexical, lexical_plus, semantic, hybrid (default). 
ner4opt = Ner4Opt(model="hybrid")

# Extract a list of dictionaries corresponding to entities found in the given problem description.
# Each dictionary holds keys for the following: 
# start (starting character index of the entity), end (ending character index of the entity)
# word (entity), entity_group (entity label) and score (confidence score for the entity)
entities = ner4opt.get_entities(problem_description)

# Output
print("Number of entities found: ", len(entities))

# Example entity output
[   
    {
        'start': 32, 
        'end': 37, 
        'word': 'total', 
        'entity_group': 'CONST_DIR', 
        'score': 0.997172257043559
    },
    {
        'start': 575, 
        'end': 583, 
        'word': 'maximize', 
        'entity_group': 'OBJ_DIR', 
        'score': 0.9982091561140413
    },
    { ... },
]
```

## Installation

Ner4Opt requires Python 3.8+ and can be installed from PyPI using `pip install ner4opt` or by building from source 

```bash
git clone https://github.com/skadio/ner4opt.git
cd ner4opt
pip install .
```

## Testing

To run tests, execute the following from the root folder:

```
python -m unittest
```

## Citation

## Citation
If you use Ner4Opt, please cite the following paper:

```
@inproceedings {ner4opt,
    title = {Ner4Opt: Named Entity Recognition for Optimization Modelling from Natural Language}
	author = {Parag Pravin Dakle, Serdar Kadıoğlu, Karthik Uppuluri, Regina Politi, Preethi Raghavan, SaiKrishna Rallabandi, Ravisutha Srinivasamurthy}
    journal = {The 20th International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research (CPAIOR 2023)},
	year = {2023},
}

```
