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

Here are the details of our [pre-trained models](https://github.com/skadio/ner4opt/blob/main/models/README.md) and the [training procedure](https://github.com/skadio/ner4opt/blob/main/models/training/README.md). Large pretrained models are hosted on [HuggingFace Models](https://huggingface.co/skadio).

## Quick Start

```python
from ner4opt import Ner4Opt

# Input optimization problem description as free-form text
problem_description = "The Notorious Desk company wants to promote a new brand of wine and wants to market it using a total market budget of $ 87,000 . To do so , the company needs to decide how much to allocate on each of its two advertising channels : ( 1 ) morning TV show and ( 2 ) social media . Each day , it costs the company $ 1,000 and $ 2000 to run advertisement spots on morning TV show and social media respectively . The expected daily reach , based on past ratings , is 15,000 viewers for each morning show spot and 30,000 internet users for a social media spot . The chief marketer knows from her experience that both channels are key to the success of the product launch . She wants to plan at least 4 but no more than 7 morning show spots . In addition , the social media spots needs to be at least 30 due to pricing tier policy . How many times should each of the media channels be used to maximize the reach of the campaign ?"

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
        'start': 108,
        'end': 114,
        'word': 'budget',
        'entity_group': 'CONST_DIR',
        'score': 0.9919970308651846
    },
    {
        'start': 120,
        'end': 126,
        'word': '87,000',
        'entity_group': 'LIMIT',
        'score': 0.9993724035912778
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
