class Constants:

    CLASSICAL = 'classical'
    CLASSICAL_PLUS = 'classical_plus'
    SEMANTIC = 'semantic'
    HYBRID = 'hybrid'

    MODELS_DIRECTORY = 'models'

    # CRF Models
    CLASSICAL_MODEL_NAME = 'classical.pkl'
    CLASSICAL_PLUS_MODEL_NAME = 'classical_plus.pkl'
    HYBRID_MODEL_NAME = 'hybrid.pkl'

    # Deep Models
    SEMANTIC_DEEP_MODEL = 'skadio/ner4opt-roberta-v1'
    HYBRID_DEEP_MODEL = 'skadio/ner4opt-roberta-v2'

    # For generating these keywords automatically, please refer our jupyter notebooks
    CONST_DIR_KEYWORDS = [
        'not want to spend more than', 'not want to invest more than', 'can not sell more than',
        'can not make more than', 'make only up to', 'can spend a total', 'can be used for', "n't make more than",
        'not make more than', 'not be greater than', 'not eat more than', 'not prepare more than', 'no less than',
        'can not exceed', 'at the most', 'maximum daily budget', 'does not exceed', 'has to collect', 'must not exceed',
        'limited the number', 'limits the number', 'a total of', 'no more than', 'has on hand', 'at lest', 'more that',
        'maximum budget', 'at most', 'not exceed', 'production capacity', 'in total', 'up to', 'minimum requirements',
        'at least', 'can buy', 'available stock', 'more than', 'less than', 'available', 'inventory', 'capacity',
        'more', 'acquired', 'has', 'invest', 'total', 'only', 'limit', 'limits', 'provide', 'have', 'competes',
        'maximum', 'receives', 'budget', 'produces', 'using', 'requirements', 'below', 'in', 'at', 'stock', 'sells',
        'minimum'
    ]

    OBJ_DIR_KEYWORDS = [
        'minimizes', 'least', 'maximum', 'maximized', 'minimizing', 'using', 'lowest', 'maximise', 'fewest',
        'overstate', 'most', 'minimize', 'high', 'maximal', 'minimal', 'minimum', 'maximize', 'minimise'
    ]

    SUBJECTS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl']

    OBJECTS = ['dobj', 'dative', 'attr', 'oprd', 'pobj', 'compound']

    ## https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    ## https://spacy.io/api/matcher
    OBJ_NAME_PATTERN = [
        {
            "LEMMA": {
                "IN": OBJ_DIR_KEYWORDS
            }
        },
        {
            "POS": {
                "IN": ["DET", "PRON", "PROPN"]
            },
            "OP": "?"
        },
        {
            "POS": {
                "IN": ["PART", "ADJ"]
            },
            "OP": "*"
        },
        {
            "POS": "NOUN",
            "OP": "+"
        },
        {
            "POS": "VERB",
            "OP": "*"
        },
        {
            "POS": "NOUN",
            "OP": "*"
        },
    ]

    # Spacy models
    # better for sentence tokenization
    SPACY_SMALL_MODEL = 'en_core_web_sm'
    # better for pos, ner, dep
    SPACY_TRANSFORMERS_MODEL = 'en_core_web_trf'

    LABELS = [
        'O', 'B-CONST_DIR', 'B-LIMIT', 'B-VAR', 'I-VAR', 'B-PARAM', 'I-PARAM', 'B-OBJ_NAME', 'I-LIMIT', 'I-CONST_DIR',
        'B-OBJ_DIR', 'I-OBJ_NAME'
    ]
