import re
from typing import Any, Type, Dict, List, Tuple

import inflect
import numpy as np
import spacy
from scipy.special import softmax
from simpletransformers.ner import NERArgs, NERModel
from spacy.training import biluo_tags_to_offsets, iob_to_biluo

from .constants import Constants

nlp = spacy.load(Constants.SPACY_SMALL_MODEL)

SpacyType: Type[Any] = List[spacy.tokens.token.Token]
SpanType: Type[Any] = type(spacy.tokens.Span)

regex_obj = re.compile(r'\S+')
pluralizer = inflect.engine()


def spacy_tokenize_sentence(sentence: str) -> str:
    """Given a sentence tokenize the text using SpaCy."""
    return " ".join([token.text for token in nlp(sentence)])


def l2_augment_sentence(sentence: str) -> Tuple[str, str]:
    """Given a sentence concatenate last two sentences to the top.

    Function returns both augmented sentence and the augmentation
    """

    spacy_sentence = nlp(sentence)

    if len(list(spacy_sentence.sents)) >= 2:
        last_two_sentences = ' '.join([item.text for item in list(spacy_sentence.sents)[-2::]])
    else:
        last_two_sentences = ' '.join([item.text for item in list(spacy_sentence.sents)[-1::]])

    augmented_sentence = last_two_sentences + " " + spacy_sentence.text
    augmentation = last_two_sentences

    return augmented_sentence, augmentation


def generate_singulars_and_plurals(phrases: List[str]) -> List[str]:
    """Generate both singulars and plurals of a given list of phrases."""

    plurals_and_singulars = []

    for phrase in phrases:

        if pluralizer.plural_noun(phrase):
            plurals_and_singulars.append(pluralizer.plural_noun(phrase))

        if pluralizer.singular_noun(phrase):
            plurals_and_singulars.append(pluralizer.singular_noun(phrase))

        # to account for inflect mistakes
        if len(phrase.split()) > 1:
            last_word = phrase.split()[-1]

            if pluralizer.plural_noun(last_word):
                plurals_and_singulars.append(' '.join(phrase.split()[:-1] + [pluralizer.plural_noun(last_word)]))

            if pluralizer.singular_noun(last_word):
                plurals_and_singulars.append(' '.join(phrase.split()[:-1] + [pluralizer.singular_noun(last_word)]))

    return plurals_and_singulars


def get_phrases_with_hyphens_and_quotes(spacy_sentence: SpacyType) -> List[SpanType]:
    """Extract all the phrases with hyphens and quotes given a spacy sentence."""

    hyphens_and_quotes = []

    matches = re.finditer(r'((?:\w+\s-\s)+\w+)|"(.+?)"|“(.+?)”', spacy_sentence.text)
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(spacy_sentence.text))}

    for m in matches:

        try:
            start = word_dict[m.start()][0]
        except KeyError:
            # in any rare scenario of mis-tokenization just continue and ignore match
            continue
        end = start + len(m.group().split())
        hyphens_and_quotes.append(spacy_sentence[start:end])

    return hyphens_and_quotes


def generate_connected_components(spacy_sentence: SpacyType) -> Dict:
    """Generates connected components in each sentence."""

    # hyphens and quotes
    hyphens_and_quotes = get_phrases_with_hyphens_and_quotes(spacy_sentence)

    # Noun Phrases
    noun_chunks = [item for item in list(spacy_sentence.noun_chunks) if len(item.text.split()) > 1]

    final_chunks = hyphens_and_quotes + noun_chunks
    final_chunks = spacy.util.filter_spans(final_chunks)

    connected_chunks = {token.i: [token.i, token.i] for token in spacy_sentence}

    for chunk in final_chunks:

        for token_idx in range(chunk.start, chunk.end):

            connected_chunks[token_idx] = [chunk.start, chunk.end]

    return connected_chunks


def load_torch_model(model_name: str, use_gpu: bool = False):
    """Load Torch model."""

    model_args = NERArgs()
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "eval_loss"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 2000
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.num_train_epochs = 11
    model_args.adafactor_beta1 = 0.9
    model_args.weight_decay = 0.01
    model_args.max_seq_length = 512
    model_args.learning_rate = 4e-5
    model_args.train_batch_size = 1
    model_args.eval_batch_size = 1
    model_args.manual_seed = 123456789
    model_args.use_cuda = use_gpu
    model_args.force_download = True

    model = NERModel("roberta", model_name, labels=Constants.LABELS, use_cuda=use_gpu, args=model_args)

    return model


def generate_feature_dictionary(features: List, token_index: int, model_name: str) -> Dict:
    """Generate feature dictionary required for CRF input."""

    features_dictionary = {
        'bias': 1.0,
    }
    for feature_name in Constants.LEXICAL_FEATURES:
        features_dictionary.update({feature_name: features[token_index][Constants.FEATURE_INDEX_MAP[feature_name]]})

    # add engineered features if lexical_plus or hybrid
    if model_name in [Constants.LEXICAL_PLUS, Constants.HYBRID]:
        for feature_name in Constants.LEXICAL_PLUS_FEATURES:
            features_dictionary.update({feature_name: features[token_index][Constants.FEATURE_INDEX_MAP[feature_name]]})

        # add transformer prediction if hybrid
        if model_name == Constants.HYBRID:
            features_dictionary.update({
                Constants.HYBRID_FEATURES[0]:
                    features[token_index][Constants.FEATURE_INDEX_MAP[Constants.HYBRID_FEATURES[0]]]
            })

    # Beginning of the sentence and End of sentence features
    if token_index == 0:
        features_dictionary.update({'BOS': True})
    if token_index == len(features) - 1:
        features_dictionary.update({'EOS': True})

    # chose a smaller window to reduce over fitting on words during training
    for window_index in [-3, -2, -1, 1, 2, 3]:
        case_1 = window_index > 0 and len(features) > token_index + window_index
        case_2 = window_index < 0 and token_index + window_index >= 0

        if case_1 or case_2:
            features_dictionary.update({
                str(window_index) + ":lower_cased_word": features[token_index + window_index][1],
                str(window_index) + ":lower_cased_word[-3:]": features[token_index + window_index][1][-3:],
                str(window_index) + ":lower_cased_word[-2:]": features[token_index + window_index][1][-2:],
                str(window_index) + ":lemma": features[token_index + window_index][4],
            })

    # longest entity phrase length in training is 6 ##
    for window_index in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        case_1 = window_index > 0 and len(features) > token_index + window_index
        case_2 = window_index < 0 and token_index + window_index >= 0
        # didn't add transformer prediction here; as it might over fit during training to that feature completely
        if case_1 or case_2:

            for feature_name in Constants.WINDOW_FEATURES_LEXICAL:
                features_dictionary.update({
                    str(window_index) + ":" + feature_name:
                        features[token_index + window_index][Constants.FEATURE_INDEX_MAP[feature_name]]
                })

            if model_name in [Constants.LEXICAL_PLUS, Constants.HYBRID]:
                for feature_name in Constants.LEXICAL_PLUS_FEATURES:
                    features_dictionary.update({
                        str(window_index) + ":" + feature_name:
                            features[token_index + window_index][Constants.FEATURE_INDEX_MAP[feature_name]]
                    })

    return features_dictionary


def _get_token_for_char(doc: SpacyType, char_idx: int) -> int:
    """Get the token id from character index."""

    for i, token in enumerate(doc):

        if char_idx > token.idx:
            continue

        if char_idx == token.idx:
            return i

        if char_idx < token.idx:
            return i - 1

    return len(doc) - 1


def _format_predictions(predictions, probabilities, crf_flag):
    """Format predictions as tuple with word, entity tag and probability."""

    predictions_tuple = []

    for prediction, probability in zip(predictions, probabilities):

        word = list(prediction.keys())[0]

        if crf_flag:
            token_probability = probability
        else:
            un_normalized_token_probabilities = probability[word]
            normalized_probabilities = list(softmax(np.mean(un_normalized_token_probabilities, axis=0)))
            token_probability = np.max(normalized_probabilities)

        predictions_tuple.append((word, prediction[word], token_probability))

    return predictions_tuple


def format_entities(sentence: str, predictions: List, probabilities: List, crf_flag: bool = True):
    """Format entity predictions into a standard format."""

    entities_formatted = []

    spacy_sentence = nlp(sentence)
    predictions_tuple = _format_predictions(predictions, probabilities, crf_flag)
    iob_tags = [item[1] for item in predictions_tuple]
    biluo_tags = iob_to_biluo(iob_tags)
    tag_offsets = biluo_tags_to_offsets(spacy_sentence, biluo_tags)

    for tag in tag_offsets:

        start_token = _get_token_for_char(spacy_sentence, tag[0])
        word_span = spacy_sentence.text[tag[0]:tag[1]]
        length_of_span = len(word_span.split())

        if length_of_span == 1:
            probabilities = [predictions_tuple[start_token][2]]
        else:
            probabilities = [item[2] for item in predictions_tuple[start_token:start_token + length_of_span]]

        entities_formatted.append({
            "start": tag[0],
            "end": tag[1],
            "word": sentence[tag[0]:tag[1]],
            "entity_group": tag[2],
            "score": np.prod(probabilities)
        })

    return entities_formatted
