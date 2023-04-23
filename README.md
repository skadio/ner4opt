# Named Entity Recognition for Optimization (Ner4Opt) Library

Given an optimization problem in natural language, the library extracts **six named entities**:

- **CONST_DIR**: Constraint direction
- **LIMIT**: Limit
- **OBJ_DIR**: Objective direction
- **OBJ_NAME**: Objective name
- **PARAM**: Parameter
- **VAR**: Variable

## Installation

To install this package, run the following command from the root folder

```bash
pip install .
```

## Quick Start

```python
# Import the Ner4Opt Library
from ner4opt import Ner4Opt

# Problem Description
problem_description = "Cautious Asset Investment has a total of $ 150,000 to manage and decides to invest it in money market fund , which yields a 2 % return as well as in foreign bonds , which gives and average rate of return of 10.2 % . Internal policies require PAI to diversify the asset allocation so that the minimum investment in money market fund is 40 % of the total investment . Due to the risk of default of foreign countries , no more than 40 % of the total investment should be allocated to foreign bonds . How much should the Cautious Asset Investment allocate in each asset so as to maximize its average return ?"

# Ner4Opt Model options: lexical, lexical_plus, semantic, hybrid. Defaults to hybrid model
ner4opt = Ner4Opt()

# Extracts a list of dictionaries corresponding to each entity in the problem description.
# Each dictionary holds keys for start (starting character index of the entity), end (ending character index of the entity), word (entity), entity_group (entity label) and score (confidence score for the entity)
entities = ner4opt.get_entities(problem_description)

# Output
print(entities)

# Showing a prettyprint of few entities for understanding
[
    {
        ...
    },
    
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
    {
        ...
    },
    
]
```

## Testing

To run tests, execute the following from the root folder:

```
python -m unittest
```
