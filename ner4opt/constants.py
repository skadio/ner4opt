class Constants:
    """Class for all constants."""

    MODELS_DIRECTORY = 'models'

    LEXICAL = 'lexical'
    LEXICAL_PLUS = 'lexical_plus'
    SEMANTIC = 'semantic'
    HYBRID = 'hybrid'

    # CRF Models
    LEXICAL_MODEL_NAME = 'lexical.pkl'
    LEXICAL_PLUS_MODEL_NAME = 'lexical_plus.pkl'
    HYBRID_MODEL_NAME = 'hybrid.pkl'

    # Deep Models
    # Roberta Model
    SEMANTIC_DEEP_MODEL = 'skadio/ner4opt-roberta-v1'
    # Roberta Model trained as part of our Hybridization experiment
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
        'O', 'B-CONST_DIR', 'I-CONST_DIR', 'B-LIMIT', 'I-LIMIT', 'B-VAR', 'I-VAR', 'B-OBJ_DIR', 'B-OBJ_NAME',
        'I-OBJ_NAME', 'B-PARAM', 'I-PARAM'
    ]

    LEXICAL_FEATURES = [
        'word', 'lower_cased_word', 'word_is_title()', 'word_is_upper()', 'lemma', 'pos_tag', 'finegrained_pos_tag',
        'dependancy_tag', 'word_shape', 'word_is_alpha()', 'word_is_stop()', 'present_in_nltk_word_list',
        'present_in_nltk_people_names', 'is_a_noun_chunk', 'gold_entity_tag', 'prepositional_chunk'
    ]
    # additional to lexical features
    LEXICAL_PLUS_FEATURES = [
        'obj_dir_keyword_is_present', 'const_dir_keyword_is_present', 'objective_name_feature_tag',
        'var_name_feature_tag'
    ]
    # additional to lexical_plus features
    HYBRID_FEATURES = ['transformer_prediction']

    FEATURE_INDEX_MAP = {
        feature: index for index, feature in enumerate(LEXICAL_FEATURES + LEXICAL_PLUS_FEATURES + HYBRID_FEATURES)
    }

    WINDOW_FEATURES_LEXICAL = [
        feature for feature in LEXICAL_FEATURES if feature not in ['word', 'lower_cased_word', 'lemma']
    ]
