import os
import pathlib
from typing import List

import joblib

from .constants import Constants
from .featurizer import Featurizer
from .utils import format_entities, generate_feature_dictionary, l2_augment_sentence, load_torch_model, spacy_tokenize_sentence


class Ner4Opt(object):
    """Extract named entities for a given optimization problem.

    The main objective of the Ner4Opt library is to generate named entities for optimization problems.
    There are six entities the current library extracts; CONST_DIR (constraint direction), LIMIT (limit),
    OBJ_DIR (objective direction), OBJ_NAME (objective name), PARAM (parameter) and VAR (variable).

    This library provides access to four main models for extracting these entities.
    `lexical` model built using lexical features alone,
    `lexical_plus` model which in addition to lexical features uses optimization scenario specific features,
    `semantic` model which uses a roberta based transformers model and
    `hybrid` model which combines all the best features of both lexical and semantic models to achieve best results.

    Attributes
    -------
    model (str) : Specifies the model to use for extracting the entities.
                  Options include `lexical`, `lexical_plus`, `semantic` and `hybrid`
                  Default is `hybrid`
    use_gpu (bool) : Specifies the model to use gpu while calculating transformer based features.
                     Default is False
    use_multiprocessing (bool) : Specifies the model to use multiprocessing while calculating transformer based features.
                                 Default is False

    Methods
    -------
    get_entities()

    Attributes
    -------
    text (str) : Sample optimization problem

    Returns
    -------

    Returns named entities of a given optimization problem. A list of dictionaries for each entity

    Keys:
        Name: start, dtype: int: Starting character index of the entity. Range is [0, len(text)]
        Name: end, dtype: int: Ending character index of the entity. Range is [0, len(text)]
        Name: word, dtype: str: Entity phrase
        Name: entity_group, dtype: str: Type of the entity [CONST_DIR, LIMIT, OBJ_DIR, OBJ_NAME, PARAM, VAR]
        Name: score, dtype: float: Defines the confidence of the prediction. Range is [0.0, 100.0]

    Example Usage
    -------------
    # Import the Ner4Opt Library
    >>> from ner4opt import Ner4Opt

    # Problem Description
    >>> problem_description = "Cautious Asset Investment has a total of $150,000 to manage and decides to invest it in money market fund, which yields a 2% return as well as in foreign bonds, which gives and average rate of return of 10.2%. Internal policies require PAI to diversify the asset allocation so that the minimum investment in money market fund is 40% of the total investment. Due to the risk of default of foreign countries, no more than 40% of the total investment should be allocated to foreign bonds. How much should the Cautious Asset Investment allocate in each asset so as to maximize its average return?"

    # Ner4Opt Model options: lexical, lexical_plus, semantic, hybrid (default).
    >>> ner4opt = Ner4Opt(model="hybrid")

    # Extracts a list of dictionaries corresponding to entities found in the given problem description.
    # Each dictionary holds keys for the following:
    # start (starting character index of the entity), end (ending character index of the entity)
    # word (entity), entity_group (entity label) and score (confidence score for the entity)
    >>> entities = ner4opt.get_entities(problem_description)

    # Output
    >>> print("Number of entities found: ", len(entities))

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

    Help
    ----
    >>> from ner4opt import Ner4Opt
    >>> Ner4Opt.__doc__
    """

    def __init__(self, model: str = Constants.HYBRID, use_gpu: bool = False, use_multiprocessing: bool = False):
        """Init Ner4Opt class."""

        _root_directory = pathlib.Path(__file__).parent.parent

        self._lexical_crf_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY,
                                                    Constants.LEXICAL_CRF_MODEL_NAME)
        self._lexical_plus_crf_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY,
                                                         Constants.LEXICAL_PLUS_CRF_MODEL_NAME)
        self._hybrid_crf_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY,
                                                   Constants.HYBRID_CRF_MODEL_NAME)

        self._model = model
        self._crf_model = None
        self._semantic_feature_extractor = None

        # Load model
        if self._model == Constants.LEXICAL:
            self._crf_model = joblib.load(self._lexical_crf_model_path)

        elif self._model == Constants.LEXICAL_PLUS:
            self._crf_model = joblib.load(self._lexical_plus_crf_model_path)

        elif self._model == Constants.SEMANTIC:
            # Model to extract semantic features
            self._semantic_feature_extractor = load_torch_model(Constants.SEMANTIC_MODEL_ROBERTA_V1,
                                                                use_gpu=use_gpu,
                                                                use_multiprocessing=use_multiprocessing)

        elif self._model == Constants.HYBRID:
            self._semantic_feature_extractor = load_torch_model(Constants.SEMANTIC_MODEL_ROBERTA_V2,
                                                                use_gpu=use_gpu,
                                                                use_multiprocessing=use_multiprocessing)
            # Our best model combines both semantic and lexical features
            self._crf_model = joblib.load(self._hybrid_crf_model_path)

        else:
            raise ValueError(
                "Invalid Model {} passed. Model name should be one of the following lexical, lexical_plus, semantic or hybrid"
                .format(self._model))

    def get_entities(self, text: str) -> List[dict]:
        """Extract Entities."""

        predicted_entities = []

        if not text:
            return predicted_entities

        # tokenize to make sure the text is in SpaCy tokenized format
        text = spacy_tokenize_sentence(text)

        augmented_sentence, augmentation = l2_augment_sentence(text)
        featurizer = Featurizer(augmented_sentence, self._model)
        tokens_to_ignore_from_augmentation = len(augmentation.split())

        if self._model == Constants.SEMANTIC:
            # use semantic features and associated probabilities to predict entity labels
            predicted_entities, probabilities = self._semantic_feature_extractor.predict([augmented_sentence], split_on_space=True)
            # ignore augmentation tokens
            predicted_entities = predicted_entities[0][tokens_to_ignore_from_augmentation::]
            probabilities = probabilities[0][tokens_to_ignore_from_augmentation::]
            # format predicted entities
            predicted_entities = format_entities(text, predicted_entities, probabilities, crf_flag=False)
        else:
            # extract all lexical features
            features = featurizer.get_features()

            if self._model == Constants.HYBRID:
                # add semantic predictions as an additional feature
                semantic_features, _ = self._semantic_feature_extractor.predict([augmented_sentence], split_on_space=True)
                semantic_features = [list(item.values())[0] for item in semantic_features[0]]

                for token_index, token_features in enumerate(features):
                    tmp_features = token_features + (semantic_features[token_index],)
                    features[token_index] = tmp_features

            # generate feature dictionary, an input required for the CRF model
            crf_model_input = []
            for feature_index in range(len(features)):
                crf_model_input.append(generate_feature_dictionary(features, feature_index, self._model))

            # predict and format
            predicted_entities = self._crf_model.predict([crf_model_input])
            predicted_entities_probabilities = self._crf_model.best_estimator_.predict_marginals([crf_model_input])

            probabilities = []
            for entity_index, entity in enumerate(predicted_entities[0]):
                probabilities.append(predicted_entities_probabilities[0][entity_index][entity])
            predicted_entities = predicted_entities[0][tokens_to_ignore_from_augmentation::]

            tokens = text.split()
            predicted_entities = [{tokens[item_index]: item} for item_index, item in enumerate(predicted_entities)]
            probabilities = probabilities[tokens_to_ignore_from_augmentation::]
            predicted_entities = format_entities(text, predicted_entities, probabilities)

        return predicted_entities
