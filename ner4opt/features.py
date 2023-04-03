import re

import nltk
import spacy
from nltk.corpus import names, wordnet, words
from nltk.stem import WordNetLemmatizer
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer

from .constants import Constants
from .utils import generate_connected_components, prune_spans, generate_singulars_and_plurals

nltk.download('all')

nlp = spacy.load("en_core_web_trf")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

nlp_small = spacy.load("en_core_web_sm")
nlp_small.tokenizer = Tokenizer(nlp_small.vocab, token_match=re.compile(r'\S+').match)

OBJ_NAME_MATCHER = Matcher(nlp.vocab)
OBJ_NAME_MATCHER.add("objective_name", [Constants.OBJ_NAME_PATTERN])

wordset = set(words.words())
nameset = set(names.words())
lemmatizer = WordNetLemmatizer()
regex_obj = re.compile(r'\S+')
special_characters = ".\+*?[^]$(){}=!<>|:-"


def generate_keyword_features(sentence, keywords, label):
    """."""
    keyword_tags = ['O' for _ in sentence.split()]
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(sentence))}
    for keyword in keywords:
        match_string = r'\b{}\b'.format(keyword.lower())
        match_result = re.finditer(match_string, sentence.lower())
        for m in match_result:
            start = word_dict[m.start()][0]
            end = start + len(m.group().split())
            for idx in range(start, end):
                if idx == start:
                    keyword_tags[idx] = "B-" + label
                else:
                    keyword_tags[idx] = "I-" + label
    return keyword_tags


def generate_entity_keyword_features_entities(spacy_sentence):
    """."""
    entity_tags = ['O' for _ in spacy_sentence]
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(spacy_sentence.text))}
    entities = spacy_sentence.ents
    for entity in entities:
        for entity_idx in range(entity.start, entity.end):
            if entity_idx == entity.start:
                entity_tags[entity_idx] = "B-" + entity.label_
            else:
                entity_tags[entity_idx] = "I-" + entity.label_
    return entity_tags


def extract_prepositional_chunks(spacy_sentence, spacy_connected_chunk, condition_on_obj_dir_keywords=True):

    prepositional_chunks = []
    for token in spacy_sentence:

        if token.pos_ == 'ADP' and token.orth_ == 'of':
            subtree_spans = [tok.i for tok in token.subtree]
            start_index = min(subtree_spans) - 1
            end_index = max(subtree_spans) + 1

            if start_index in spacy_connected_chunk:
                start_index = spacy_connected_chunk[start_index][0]
            if end_index in spacy_connected_chunk:
                end_index = spacy_connected_chunk[end_index][1]

            if start_index < 0:
                start_index = 0

            if end_index == len(spacy_sentence):
                end_index -= 1

            if condition_on_obj_dir_keywords:
                if spacy_sentence[start_index - 1].lemma_ in Constants.OBJ_DIR_KEYWORDS:
                    # push end_index to search for any missing verb nouns
                    while spacy_sentence[end_index].pos_ in ['VERB', 'NOUN']:
                        end_index += 1
                        if end_index == len(spacy_sentence):
                            end_index -= 1
                            break
                    prepositional_chunks.append(spacy_sentence[start_index:end_index])
            else:
                # push end_index to search for any missing verb nouns
                while spacy_sentence[end_index].pos_ in ['VERB', 'NOUN']:
                    end_index += 1
                    if end_index == len(spacy_sentence):
                        end_index -= 1
                        break
                prepositional_chunks.append(spacy_sentence[start_index:end_index])

    return prepositional_chunks


def extract_objective_name_patterns(spacy_sentence, spacy_sentence_connected_chunk):
    """Extract Objective name spans by finding patterns."""
    matches = OBJ_NAME_MATCHER(spacy_sentence)
    spans = []
    for _, start, end in matches:
        updated_start = spacy_sentence_connected_chunk[start + 1][0]
        updated_end = spacy_sentence_connected_chunk[start + 1][1]
        if updated_start == updated_end:
            spans.append(spacy_sentence[start + 1:end])
        else:
            spans.append(spacy_sentence[start + 1:updated_end])
    spans = spacy.util.filter_spans(spans)
    return spans


def extract_objective_name_chunks(spacy_sentence, spacy_sentence_connected_chunk):
    """Function extracts objective name chunks from each sentence."""
    objective_names = []
    prep_chunks = extract_prepositional_chunks(spacy_sentence, spacy_sentence_connected_chunk)
    objective_name_patterns = extract_objective_name_patterns(spacy_sentence, spacy_sentence_connected_chunk)
    final_spans = prep_chunks + objective_name_patterns
    final_spans = spacy.util.filter_spans(final_spans)
    final_spans += prune_spans(final_spans, spacy_sentence)
    for span in final_spans:
        for tok in span:
            if tok.dep_ in Constants.OBJECTS:
                objective_names += [tok.text]
    objective_names += [val.text for val in final_spans]
    if not objective_names:
        final_sentence = list(spacy_sentence.sents)[-1]
        for token in final_sentence:
            lefts = list(token.lefts)
            if token.pos_ in ['VERB', 'ADJ'] and token.lemma_ in OBJ_DIR_KEYWORDS:
                objective_names += [tok.text for tok in lefts if tok.dep_ in Constants.SUBJECTS and tok.pos_ == 'NOUN']
    plurals_and_singulars = generate_singulars_and_plurals(objective_names)
    objective_names += plurals_and_singulars
    objective_names = list(set(objective_names))
    objective_names.sort(key=lambda x: len(x.split()), reverse=True)
    return objective_names


def get_objective_name_spans(sentence):
    """."""
    # Spacify the sentence
    spacy_sentence = nlp(sentence)
    # Get connected chunks
    spacy_connected_chunk = generate_connected_components(spacy_sentence)

    spans = ['O' for _ in spacy_sentence]
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(spacy_sentence.text))}
    objective_names = extract_objective_name_chunks(spacy_sentence, spacy_connected_chunk)
    for phrase in objective_names:
        if any(char in special_characters for char in phrase):
            match_string = r'{}'.format(re.escape(phrase.lower()))
        else:
            match_string = r'\b{}\b'.format(phrase.lower())
        match_result = re.finditer(match_string, spacy_sentence.text.lower())
        for m in match_result:
            start = word_dict[m.start()][0]
            end = start + len(m.group().split())
            for idx in range(start, end):
                if spans[idx] == 'O':
                    if idx == start:
                        spans[idx] = "B-OBJ_NAME"
                    else:
                        spans[idx] = "I-OBJ_NAME"
    return spans


# VAR Name Span Feature


# Try to maximize coverage; false positives will be taken care of when combined with other features
def get_conjuncting_chunks(spacy_sentence, spacy_connected_chunk):

    ## get conjuncting noun chunks and prepositional chunks ##

    phrases = set()
    prep_subtrees = extract_prepositional_chunks(spacy_sentence,
                                                 spacy_connected_chunk,
                                                 condition_on_obj_dir_keywords=False)
    noun_chunks_list = list(spacy_sentence.noun_chunks)

    for nc_index, nc in enumerate(noun_chunks_list):
        case_one = nc.conjuncts
        case_two = spacy_sentence[nc.start - 1].conjuncts
        case_three = nc[0].conjuncts
        if case_one or case_two or case_three:
            phrase_one = nc
            if not nc.conjuncts:
                if nc[0].conjuncts:
                    phrase_one = nc[0]
                    phrase_two = nc[0].conjuncts[0]
                else:
                    if not spacy_sentence[nc.start - 1].conjuncts[0].i < nc.start:
                        phrase_two = noun_chunks_list[nc_index - 1]
                    else:
                        if nc_index + 1 < len(noun_chunks_list):
                            phrase_two = noun_chunks_list[nc_index + 1]
            else:
                if nc.conjuncts[0].i < nc.start:
                    phrase_two = noun_chunks_list[nc_index - 1]
                else:
                    phrase_two = noun_chunks_list[nc_index + 1]

            phrase_one_bool = False
            phrase_two_bool = False
            for keyword in Constants.CONST_DIR_KEYWORDS:
                if keyword in phrase_one.text.split():
                    phrase_one_bool = True
                    break
            for keyword in Constants.CONST_DIR_KEYWORDS:
                if keyword in phrase_two.text.split():
                    phrase_two_bool = True
                    break

            if len(phrase_one.text.split()) == 1:
                phrases.add(phrase_one.text)
            else:
                if not phrase_one_bool and phrase_one[0].pos_ != 'NUM':
                    if phrase_one[0].pos_ == 'DET':
                        phrase_one = phrase_one[1::]
                    phrases.add(phrase_one.text)

            if len(phrase_two.text.split()) == 1:
                phrases.add(phrase_two.text)
            else:
                if not phrase_two_bool and phrase_two[0].pos_ != 'NUM':
                    if phrase_two[0].pos_ == 'DET':
                        phrase_two = phrase_two[1::]
                    phrases.add(phrase_two.text)

    ## add all phrases connected via hyphens and quotes
    hyphens_and_quotes = []
    matches = re.finditer(r'((?:\w+\s-\s)+\w+)|"(.+?)"|“(.+?)”', spacy_sentence.text)
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(spacy_sentence.text))}
    for m in matches:
        try:
            start = word_dict[m.start()][0]
        except:
            continue
        # start = word_dict[m.start()][0]
        end = start + len(m.group().split())
        hyphens_and_quotes.append(spacy_sentence[start:end].text)

    phrases = list(set(list(phrases) + hyphens_and_quotes))

    ## add all other missing noun chunks which start with conjuncting items
    new_phrases = []
    chunk_texts = []
    for val in noun_chunks_list:
        if val[0].pos_ == 'DET':
            chunk_texts.append(val[1::].text)
        else:
            chunk_texts.append(val.text)
    for chunk in chunk_texts:
        lowered_chunk = chunk.lower()
        for item in phrases:
            if lowered_chunk.startswith(item.lower()):
                new_phrases.append(lowered_chunk)
    phrases = list(set(phrases + new_phrases))

    ## add all prepositional chunks which end with conjuncting items
    prep_conjuncts = []
    for tree in prep_subtrees:
        const_dir_presence = False
        pos_tags = [tok.pos_ for tok in tree]
        for keyphrase in Constants.CONST_DIR_KEYWORDS:
            if keyphrase in tree.text:
                const_dir_presence = True
                break
        if (not const_dir_presence) and ('NUM' not in pos_tags):
            for phrase_var in phrases:
                if tree.text.endswith(phrase_var):
                    if tree[0].pos_ == 'DET':
                        tree = tree[1::]
                    prep_conjuncts.append(tree.text)
    phrases = phrases + prep_conjuncts

    ## add all singulars and plurals
    singulars_and_plurals = []
    for val in phrases:
        singulars_and_plurals += generate_singulars_and_plurals([val])
    phrases = list(set(phrases + singulars_and_plurals))
    return phrases


def get_var_name_spans(sentence):
    # Spacify the sentence
    spacy_sentence = nlp(sentence)
    # Get connected chunks
    spacy_connected_chunk = generate_connected_components(spacy_sentence)
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(spacy_sentence.text))}
    spans = ['O' for tok in spacy_sentence]
    final_conjuncts = get_conjuncting_chunks(spacy_sentence, spacy_connected_chunk)
    final_conjuncts.sort(key=lambda x: len(x.split()), reverse=True)
    for phrase in final_conjuncts:
        if any(char in special_characters for char in phrase):
            match_string = r'{}'.format(re.escape(phrase.lower()))
        else:
            match_string = r'\b{}\b'.format(phrase.lower())

        match_result = re.finditer(match_string, spacy_sentence.text.lower())

        for m in match_result:
            try:
                start = word_dict[m.start()][0]
            except:
                continue
            end = start + len(m.group().split())
            for idx in range(start, end):
                if spans[idx] == 'O':
                    if idx == start:
                        spans[idx] = "B-VAR"
                    else:
                        spans[idx] = "I-VAR"
    return spans


def get_prepositional_chunks(spacy_sentence, spacy_connected_chunk):

    prep_chunk_span = ['O' for _ in spacy_sentence]
    prep_subtrees = extract_prepositional_chunks(spacy_sentence,
                                                 spacy_connected_chunk,
                                                 condition_on_obj_dir_keywords=False)
    for subtree in prep_subtrees:
        start = subtree.start
        end = start + len(subtree.text.split())
        for idx in range(start, end):
            if idx == start:
                prep_chunk_span[idx] = "B-PREP_CHUNK"
            else:
                prep_chunk_span[idx] = "I-PREP_CHUNK"

    return prep_chunk_span


def mark_keywords(sentence, keywords, label):
    keyword_tags = ['O' for _ in sentence.split()]
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(sentence))}
    for keyword in keywords:
        match_string = r'\b{}\b'.format(keyword.lower())
        match_result = re.finditer(match_string, sentence.lower())
        for m in match_result:
            try:
                start = word_dict[m.start()][0]
            except:
                continue
            # start = word_dict[m.start()][0]
            end = start + len(m.group().split())
            for idx in range(start, end):
                if idx == start:
                    keyword_tags[idx] = "B-" + label
                else:
                    keyword_tags[idx] = "I-" + label
    return keyword_tags


def mark_entities(spacy_sentence):
    entity_tags = ['O' for _ in spacy_sentence]
    word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(spacy_sentence.text))}
    entities = spacy_sentence.ents
    for entity in entities:
        for entity_idx in range(entity.start, entity.end):
            if entity_idx == entity.start:
                entity_tags[entity_idx] = "B-" + entity.label_
            else:
                entity_tags[entity_idx] = "I-" + entity.label_
    return entity_tags


def get_linguistic_features(spacy_sentence):
    """."""

    noun_chunk_labels = ['O' for _ in spacy_sentence]
    for chunk in spacy_sentence.noun_chunks:
        for chunk_idx in range(chunk.start, chunk.end):
            if noun_chunk_labels[chunk_idx] == 'O':
                if chunk_idx == chunk.start:
                    noun_chunk_labels[chunk_idx] = "B-NOUN_CHUNK"
                else:
                    noun_chunk_labels[chunk_idx] = "I-NOUN_CHUNK"

    entity_tags = mark_entities(spacy_sentence)

    features = []
    for token_index, token in enumerate(spacy_sentence):
        lower_case_token = token.text.lower()
        in_nltk_words = lower_case_token in wordset
        in_nltk_people_names = token.text in nameset

        features.append((token.text, lower_case_token, token.text.istitle(), token.text.isupper(), token.lemma_,
                         token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, in_nltk_words,
                         in_nltk_people_names, noun_chunk_labels[token_index], entity_tags[token_index]))

    return features


def get_features(sentence):
    """."""
    # Spacify the sentence
    spacy_sentence = nlp(sentence)
    # Get connected chunks
    spacy_connected_chunk = generate_connected_components(spacy_sentence)

    # get common features for all models
    linguistic_features = get_linguistic_features(spacy_sentence)
    prepositional_chunks = get_prepositional_chunks(spacy_sentence, spacy_connected_chunk)

    # Concatenate features
    common_features = []
    for token_index, token_features in enumerate(linguistic_features):
        tmp_features = token_features + (prepositional_chunks[token_index],)
        common_features.append(tmp_features)

    return common_features
