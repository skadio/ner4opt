# Named Entity Recognition for Optimization (Ner4Opt) Library

Given an optimization problem in natural language, the Ner4Opt library extracts **six named entities**:

- **CONST_DIR**: Constraint direction
- **LIMIT**: Limit
- **OBJ_DIR**: Objective direction
- **OBJ_NAME**: Objective name
- **PARAM**: Parameter
- **VAR**: Variable

## Quick Start

```python
# Import the Ner4Opt Library
from ner4opt import Ner4Opt

# Problem Description
problem_description = "Cautious Asset Investment has a total of $150,000 to manage and decides to invest it in money market fund, which yields a 2% return as well as in foreign bonds, which gives and average rate of return of 10.2%. Internal policies require PAI to diversify the asset allocation so that the minimum investment in money market fund is 40% of the total investment. Due to the risk of default of foreign countries, no more than 40% of the total investment should be allocated to foreign bonds. How much should the Cautious Asset Investment allocate in each asset so as to maximize its average return?"

# Ner4Opt Model options: lexical, lexical_plus, semantic, hybrid (default). 
ner4opt = Ner4Opt(model="hybrid")

# Extracts a list of dictionaries corresponding to entities found in the given problem description.
# Each dictionary holds keys for the following: 
# start (starting character index of the entity), end (ending character index of the entity)
# word (entity), entity_group (entity label) and score (confidence score for the entity)
entities = ner4opt.get_entities(problem_description)

# Output
print("Number of entities found: ", len(entities))

# Example output
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
