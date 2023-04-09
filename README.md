# Named Entity Recognition for Optimization (Ner4Opt) Library

Given an optimization problem in natural language, the library extracts **six named entities**:

- **CONST_DIR** (constraint direction)
- **LIMIT** (limit)
- **OBJ_DIR** (objective direction) 
- **OBJ_NAME** (objective name)
- **PARAM** (parameter)
- **VAR** (variable)

## Installation

```bash
pip install . 
```

## Quick Start

```python
from ner4opt import Ner4Opt
example_problem = "Cautious Asset Investment has a total of $ 150,000 to manage and decides to invest it in money market fund , which yields a 2 % return as well as in foreign bonds , which gives and average rate of return of 10.2 % . Internal policies require PAI to diversify the asset allocation so that the minimum investment in money market fund is 40 % of the total investment . Due to the risk of default of foreign countries , no more than 40 % of the total investment should be allocated to foreign bonds . How much should the Cautious Asset Investment allocate in each asset so as to maximize its average return ?"
ner4opt = Ner4Opt('hybrid')  # select a model
entities = ner4opt.get_entities(example_problem)  # get entities
```

## Testing

To run tests, execute:

```
python -m unittest
```
