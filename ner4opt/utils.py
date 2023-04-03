import re
from typing import Any, Dict, List, Tuple, Type

import inflect
import numpy as np
import pandas as pd
import spacy
from scipy.special import softmax
from simpletransformers.ner import NERArgs, NERModel
from spacy.tokenizer import Tokenizer
from spacy.training import biluo_tags_to_offsets, iob_to_biluo
from tqdm import tqdm

from .constants import Constants

nlp = spacy.load(Constants.SPACY_SMALL_MODEL)
NlpType: Type[Any] = type(nlp)
SpanType: Type[Any] = type(spacy.tokens.Span)

regex_obj = re.compile(r'\S+')
pluralizer = inflect.engine()


def read_file_to_list(filename: str) -> List[str]:
    """Reads a filename and returns a list of lines in the filename."""

    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    return lines


def iob_to_sentences_and_labels(data: pd.DataFrame) -> Tuple[List[str], List[List[str]]]:
    """Given data in an IOB format convert data into sentences and corresponding token labels."""

    sentences = []
    labels = []

    tmp_sent = ''
    tmp_label = []

    for item_index, item in tqdm(enumerate(data)):

        # iterate until you either find a empty line break or end of the data
        if item == '' or item_index == len(data) - 1:

            sentences.append(tmp_sent.lstrip())
            labels.append(tmp_label)

            # reset
            tmp_sent = ''
            tmp_label = []
        else:
            tmp_sent = tmp_sent + " " + item.split("\t")[0]
            tmp_label.append(item.split('\t')[-1])

    return sentences, labels


def spacify_sentences(sentences: List[str], spacy_handle: NlpType) -> List[NlpType]:
    """Given a list of sentences convert them into spacy objects."""
    return [spacy_handle(sentence) for sentence in sentences]


def l2_augment_sentence(sentence):
    """Given a list of spacy objects concatenate last two sentences to the top for each object."""

    spacy_sentence = nlp(sentence)

    if len(list(spacy_sentence.sents)) >= 2:
        last_two_sentences = ' '.join([item.text for item in list(spacy_sentence.sents)[-2::]])
    else:
        last_two_sentences = ' '.join([item.text for item in list(spacy_sentence.sents)[-1::]])

    augmented_sentence = last_two_sentences + " " + spacy_sentence.text
    augmentation = last_two_sentences

    return augmented_sentence, augmentation


def generate_connected_components(spacy_sentence: NlpType) -> Dict:
    """Function generates connected components in each sentence."""

    # hyphens and quotes
    hyphens_and_quotes = []
    matches = re.finditer(r'((?:\w+\s-\s)+\w+)|"(.+?)"|“(.+?)”', spacy_sentence.text)
    regex_obj = re.compile(r'\S+')
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(spacy_sentence.text))}

    for m in matches:
        start = word_dict[m.start()][0]
        end = start + len(m.group().split())
        hyphens_and_quotes.append(spacy_sentence[start:end])

    # Noun Phrases
    noun_chunks = [item for item in list(spacy_sentence.noun_chunks) if len(item.text.split()) > 1]

    final_chunks = hyphens_and_quotes + noun_chunks
    final_chunks = spacy.util.filter_spans(final_chunks)

    connected_chunks = {token.i: [token.i, token.i] for token in spacy_sentence}

    for chunk in final_chunks:
        for token_idx in range(chunk.start, chunk.end):
            connected_chunks[token_idx] = [chunk.start, chunk.end]

    return connected_chunks


def generate_singulars_and_plurals(phrases: List[str]) -> List[str]:
    """Function to generate both singulars and plurals of a given list of phrases"""

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


def prune_spans(spans: List[SpanType], spacy_sentence: NlpType) -> List[SpanType]:
    """Function prunes spans to ignore certain determiners, adjectives and prepositions."""

    pruned_spans = []

    for span in spans:

        if span[0].pos_ in ['DET', 'ADJ']:
            pruned_spans += [spacy_sentence[span[1].i:span[-1].i + 1]]

        if span[0].pos_ == 'DET' and span[1].pos_ == 'ADJ':
            pruned_spans += [spacy_sentence[span[2].i:span[-1].i + 1]]

        prepositional_indices = [tok.i for tok in span if tok.pos_ == 'ADP']

        # only consider "of" prepositional phrases and ignore rest of adpositions
        # We only consider "of" (as it is most seen in training)

        if len(prepositional_indices) > 1:

            for idx in prepositional_indices:
                if spacy_sentence[idx].orth_ == 'of':
                    rights_indices = [item.i for item in spacy_sentence[idx].rights]
                    if rights_indices:
                        pruned_spans += [spacy_sentence[idx + 1:max(rights_indices) + 1]]

    return pruned_spans


def get_feature_dict(feature, token_index, model_name):
    """."""
    word = feature[token_index][0]
    features_dictionary = {
        'bias': 1.0,
        'word': feature[token_index][0],
        'lower_cased_word': feature[token_index][1],
        'word_is_title()': feature[token_index][2],
        'word_is_upper()': feature[token_index][3],
        'lemma': feature[token_index][4],
        'pos_tag': feature[token_index][5],
        'finegrained_pos_tag': feature[token_index][6],
        'dependancy_tag': feature[token_index][7],
        'word_shape': feature[token_index][8],
        'word_is_alpha()': feature[token_index][9],
        'word_is_stop()': feature[token_index][10],
        'present_in_nltk_word_list': feature[token_index][11],
        'present_in_nltk_people_names': feature[token_index][12],
        'is_a_noun_chunk': feature[token_index][13],
        'gold_entity_tag': feature[token_index][14],
        'prepositional_chunk': feature[token_index][15],
    }

    if model_name == Constants.HYBRID:
        features_dictionary.update({
            'obj_dir_keyword_is_present': feature[token_index][16],
            'const_dir_keyword_is_present': feature[token_index][17],
            'objective_name_feature_tag': feature[token_index][18],
            'var_name_feature_tag': feature[token_index][19],
            'transformer_prediction': feature[token_index][20]
        })

    if token_index == 0:
        features_dictionary.update({'BOS': True})
    if token_index == len(feature) - 1:
        features_dictionary.update({'EOS': True})

    # chose a smaller window to reduce overfitting on words
    for window_index in [-3, -2, -1, 1, 2, 3]:
        case_1 = window_index > 0 and len(feature) > token_index + window_index
        case_2 = window_index < 0 and token_index + window_index >= 0

        if case_1 or case_2:
            features_dictionary.update({
                str(window_index) + ":lower_cased_word": feature[token_index + window_index][1],
                str(window_index) + ":lower_cased_word[-3:]": feature[token_index + window_index][1][-3:],
                str(window_index) + ":lower_cased_word[-2:]": feature[token_index + window_index][1][-2:],
                str(window_index) + ":lemma": feature[token_index + window_index][4],
            })

    ## longest entity phrase length in training is 6 ##
    for window_index in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        case_1 = window_index > 0 and len(feature) > token_index + window_index
        case_2 = window_index < 0 and token_index + window_index >= 0

        if case_1 or case_2:
            features_dictionary.update({
                str(window_index) + ':word_is_title()': feature[token_index + window_index][2],
                str(window_index) + ':word_is_upper()': feature[token_index + window_index][3],
                str(window_index) + ':pos_tag': feature[token_index + window_index][5],
                str(window_index) + ':finegrained_pos_tag': feature[token_index + window_index][6],
                str(window_index) + ':dependancy_tag': feature[token_index + window_index][7],
                str(window_index) + ':word_shape': feature[token_index + window_index][8],
                str(window_index) + ':word_is_alpha()': feature[token_index + window_index][9],
                str(window_index) + ':word_is_stop()': feature[token_index + window_index][10],
                str(window_index) + ':present_in_nltk_word_list': feature[token_index + window_index][11],
                str(window_index) + ':present_in_nltk_people_names': feature[token_index + window_index][12],
                str(window_index) + ':is_a_noun_chunk': feature[token_index + window_index][13],
                str(window_index) + ':gold_entity_tag': feature[token_index + window_index][14],
                str(window_index) + ':prepositional_chunk': feature[token_index + window_index][15],
                # didn't add transformer prediction here; as it might overfit to that feature completely
            })

            if model_name == Constants.HYBRID:
                features_dictionary.update({
                    str(window_index) + ':obj_dir_keyword_is_present': feature[token_index + window_index][16],
                    str(window_index) + ':const_dir_keyword_is_present': feature[token_index + window_index][17],
                    str(window_index) + ':objective_name_feature_tag': feature[token_index + window_index][18],
                    str(window_index) + ':var_name_feature_tag': feature[token_index + window_index][19],
                })

    return features_dictionary


def load_torch_model(model_name, use_gpu):
    """."""
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


def get_token_for_char(doc, char_idx):
    """Get the token id from character index."""
    for i, token in enumerate(doc):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return i
        if char_idx < token.idx:
            return i - 1
    return len(doc) - 1


def validate_and_format_entities(sentence, predictions, probabilities, crf=True):
    """Format predictions into spacy display formar."""
    doc = nlp(sentence)
    bert_predictions = []
    iob_tags = []
    tags_formatted = []

    for prediction, probability in zip(predictions, probabilities):
        word = list(prediction.keys())[0]
        if crf:
            prob = probability
        else:
            probas = probability[word]
            normalized_probas = list(softmax(np.mean(probas, axis=0)))
            prob = np.max(normalized_probas)
        bert_predictions.append((word, prediction[word], prob))
        iob_tags.append(prediction[word])

    biluo_tags = iob_to_biluo(iob_tags)
    tags = biluo_tags_to_offsets(doc, biluo_tags)

    for tag in tags:
        start_token = get_token_for_char(doc, tag[0])
        word_span = doc.text[tag[0]:tag[1]]
        length_of_span = len(word_span.split())
        if length_of_span == 1:
            probs = [bert_predictions[start_token][2]]
        else:
            probs = [item[2] for item in bert_predictions[start_token:start_token + length_of_span]]
        tags_formatted.append({
            "start": tag[0],
            "end": tag[1],
            "word": sentence[tag[0]:tag[1]],
            "entity_group": tag[2],
            "score": np.prod(probs)
        })
    return tags_formatted
