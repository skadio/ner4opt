import re
from typing import Any, Type, List, Tuple

import spacy
from nltk.corpus import names, words
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer

from .constants import Constants
from .utils import generate_connected_components, generate_singulars_and_plurals, get_phrases_with_hyphens_and_quotes

nlp = spacy.load(Constants.SPACY_TRANSFORMERS_MODEL)
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

OBJ_NAME_MATCHER = Matcher(nlp.vocab)
OBJ_NAME_MATCHER.add("objective_name", [Constants.OBJ_NAME_PATTERN])

word_set = set(words.words())
name_set = set(names.words())
regex_obj = re.compile(r'\S+')
special_characters = ".\+*?[^]$(){}=!<>|:-"

SpanType: Type[Any] = type(spacy.tokens.Span)


class Featurizer(object):
    """Extract different features given an optimization problem.

    There are five main types of features this class extracts

    1. Linguistic Features:
    This section of features explores all the grammatical properties which would help disambiguate these entities.
    For instance, OBJ_DIR is often a verb, with last three characters ending in 'ize'.
    LIMIT and PARAM are some form of numerical entities. Both OBJ_NAME and VAR could be a noun chunk.
    OBJ_NAME sometimes is a direct object / subject in a sentence.

    2. Gazetteer Features:
    This section of features incorporates the domain knowledge we have using Gazetteers.

    3. Prepositional Chunks:
    This section extracts all prepositional chunks which can either be an objective name or a variable given context.

    4. Features for Objective:
    An objective in an optimization problem tends to show the following signatures,
    OBJ_NAME is often OBJ_DIR followed by
        - Noun Phrase (maximize my total profit)
        - Prepositional Phrase (maximize the number of batches of cookies)
        - Verb Phrase (minimize the amount of blood pressure reducing medicine)
    This section extracts objective name finding the above signatures

    5. Features for Variables:
    A variable in an optimization problem tends to show the following signatures,
    Variables are often conjuncting
        - Noun chunks
        - Prepositional chunks
        - Phrases connected by hyphens
        - Phrases connected by quotes
    This section extracts variables finding the above signatures

    Attributes
    -------
    sequence (str) : Optimization sequence on which features are to be extracted
    model_name (str) : Specifies which features to extract based on the model type

    Methods
    -------
    get_linguistic_features()

    Returns linguistic features for each token in the sentence as a list of tuples.

    get_gazetteer_features()

    Attributes
    -------
    keywords (list) : List of keywords/key-phrases to use as a gazetteer
    label (str) : Entity type label

    Returns a list mapping each token to an entity label, if found in the gazetteer

    get_prepositional_chunk_spans()

    Returns a list mapping each token to a prepositional label, if the token is part of a prepositional chunk

    get_objective_name_spans()

    Returns a list mapping each token to an objective name entity label,
    if the tokens satisfy the above criterion mentioned

    get_variable_spans()

    Returns a list mapping each token to a variable entity label, if the tokens satisfy the above criterion mentioned

    get_features()

    An orchestrator combining different features according to the model type

    Returns all the features as a list of tuples

    """

    @staticmethod
    def _filter_noun_chunk_conjuncts(conjuncts: List[SpanType]) -> List[str]:
        """Filters noun chunk conjuncts to exclude certain signatures.

        Removes conjuncts with any intersection with CONST_DIR keyword
        Removes conjuncts starting with numerals
        Removes determiners from the conjuncts
        """

        filtered_conjuncts = []

        for conjunct in conjuncts:

            token_intersection = False

            for keyword in Constants.CONST_DIR_KEYWORDS:

                if keyword in conjunct.text.split():
                    token_intersection = True
                    break

            if len(conjunct.text.split()) == 1:
                filtered_conjuncts.append(conjunct.text)
            else:
                if not token_intersection and conjunct[0].pos_ != 'NUM':
                    if conjunct[0].pos_ == 'DET':
                        conjunct = conjunct[1::]
                    filtered_conjuncts.append(conjunct.text)

        return filtered_conjuncts

    @staticmethod
    def _get_prepositional_chunk_conjuncts(prepositional_subtrees: List[SpanType], conjuncts: List[str]) -> List[str]:
        """Extract conjuncting prepositional chunks."""

        prepositional_conjuncts = []

        for tree in prepositional_subtrees:

            const_dir_presence = False
            pos_tags = [token.pos_ for token in tree]

            for key_phrase in Constants.CONST_DIR_KEYWORDS:
                if key_phrase in tree.text:
                    const_dir_presence = True
                    break

            if (not const_dir_presence) and ('NUM' not in pos_tags):

                for conjunct in conjuncts:

                    if tree.text.endswith(conjunct):
                        if tree[0].pos_ == 'DET':
                            tree = tree[1::]
                        prepositional_conjuncts.append(tree.text)

        return prepositional_conjuncts

    def __init__(self, sequence, model_name):
        """Init Featurizer class."""

        self._model_name = model_name

        self._sequence = sequence
        self._word_dict = {m.start(0): (i, m.group(0)) for i, m in enumerate(regex_obj.finditer(self._sequence))}
        self._spacy_sequence = nlp(self._sequence)
        self._spacy_connected_components = generate_connected_components(self._spacy_sequence)

    def _extract_prepositional_chunks(self, condition_on_obj_dir_keywords: bool = True) -> List[SpanType]:
        """Extract prepositional chunks conditioned on OBJ_DIR keywords."""

        prepositional_chunks = []

        for token in self._spacy_sequence:

            if token.pos_ == 'ADP' and token.orth_ == 'of':
                subtree_spans = [tok.i for tok in token.subtree]
                start_index = min(subtree_spans) - 1
                end_index = max(subtree_spans) + 1

                if start_index in self._spacy_connected_components:
                    start_index = self._spacy_connected_components[start_index][0]

                if start_index < 0:
                    start_index = 0

                if end_index in self._spacy_connected_components:
                    end_index = self._spacy_connected_components[end_index][1]

                if end_index == len(self._spacy_sequence):
                    end_index -= 1

                if condition_on_obj_dir_keywords:
                    if self._spacy_sequence[start_index - 1].lemma_ in Constants.OBJ_DIR_KEYWORDS:
                        # push end_index to search for any missing verb nouns
                        while self._spacy_sequence[end_index].pos_ in ['VERB', 'NOUN']:
                            end_index += 1
                            if end_index == len(self._spacy_sequence):
                                end_index -= 1
                                break
                        prepositional_chunks.append(self._spacy_sequence[start_index:end_index])
                else:
                    # push end_index to search for any missing verb nouns
                    while self._spacy_sequence[end_index].pos_ in ['VERB', 'NOUN']:
                        end_index += 1
                        if end_index == len(self._spacy_sequence):
                            end_index -= 1
                            break
                    prepositional_chunks.append(self._spacy_sequence[start_index:end_index])

        return prepositional_chunks

    def _extract_objective_name_patterns(self) -> List[SpanType]:
        """Extract Objective name spans by finding patterns."""

        matches = OBJ_NAME_MATCHER(self._spacy_sequence)

        spans = []

        for _, start, end in matches:

            updated_start = self._spacy_connected_components[start + 1][0]
            updated_end = self._spacy_connected_components[start + 1][1]

            if updated_start == updated_end:
                spans.append(self._spacy_sequence[start + 1:end])
            else:
                spans.append(self._spacy_sequence[start + 1:updated_end])

        spans = spacy.util.filter_spans(spans)

        return spans

    def _prune_spans(self, spans: List[SpanType]) -> List[SpanType]:
        """Prune spans to ignore determiners, adjectives and adpositions."""

        pruned_spans = []

        for span in spans:

            if span[0].pos_ in ['DET', 'ADJ']:
                pruned_spans += [self._spacy_sequence[span[1].i:span[-1].i + 1]]

            if span[0].pos_ == 'DET' and span[1].pos_ == 'ADJ':
                pruned_spans += [self._spacy_sequence[span[2].i:span[-1].i + 1]]

            prepositional_indices = [tok.i for tok in span if tok.pos_ == 'ADP']

            # only consider "of" prepositional phrases and ignore rest of adpositions
            # We only consider "of" (as it is most seen in training)

            if len(prepositional_indices) > 1:

                for idx in prepositional_indices:

                    if self._spacy_sequence[idx].orth_ == 'of':
                        rights_indices = [item.i for item in self._spacy_sequence[idx].rights]

                        if rights_indices:
                            pruned_spans += [self._spacy_sequence[idx + 1:max(rights_indices) + 1]]

        return pruned_spans

    def _extract_objective_name_chunks(self) -> List[str]:
        """Extract objective name chunks."""

        objective_names = []

        prepositional_chunks = self._extract_prepositional_chunks()
        objective_name_patterns = self._extract_objective_name_patterns()

        objective_name_spans = prepositional_chunks + objective_name_patterns
        objective_name_spans = spacy.util.filter_spans(objective_name_spans)
        objective_name_spans += self._prune_spans(objective_name_spans)

        for span in objective_name_spans:

            for token in span:

                if token.dep_ in Constants.OBJECTS:
                    objective_names += [token.text]

        objective_names += [val.text for val in objective_name_spans]

        if not objective_names:

            last_sentence = list(self._spacy_sequence.sents)[-1]

            for token in last_sentence:

                lefts = list(token.lefts)

                if token.pos_ in ['VERB', 'ADJ'] and token.lemma_ in Constants.OBJ_DIR_KEYWORDS:
                    objective_names += [
                        tok.text for tok in lefts if tok.dep_ in Constants.SUBJECTS and tok.pos_ == 'NOUN'
                    ]

        plurals_and_singulars = generate_singulars_and_plurals(objective_names)

        objective_names += plurals_and_singulars
        objective_names = list(set(objective_names))
        objective_names.sort(key=lambda x: len(x.split()), reverse=True)

        return objective_names

    def _mark_entities(self) -> List[str]:
        """Mark each token with its corresponding gold entity label."""

        entity_tags = ['O' for _ in self._spacy_sequence]
        entities = self._spacy_sequence.ents

        for entity in entities:

            for entity_index in range(entity.start, entity.end):

                if entity_index == entity.start:
                    entity_tags[entity_index] = "B-" + entity.label_
                else:
                    entity_tags[entity_index] = "I-" + entity.label_

        return entity_tags

    def get_gazetteer_features(self, keywords: List[str], label: str) -> List[str]:
        """Extract gazetteer features by marking keywords found with a given label type."""

        keyword_tags = ['O' for _ in self._sequence.split()]

        for keyword in keywords:

            match_string = r'\b{}\b'.format(keyword.lower())
            match_result = re.finditer(match_string, self._sequence.lower())

            for m in match_result:

                start = self._word_dict[m.start()][0]
                end = start + len(m.group().split())

                for idx in range(start, end):
                    if idx == start:
                        keyword_tags[idx] = "B-" + label
                    else:
                        keyword_tags[idx] = "I-" + label

        return keyword_tags

    def get_objective_name_spans(self) -> List[str]:
        """Identify objective name spans in a given sequence."""

        spans = ['O' for _ in self._spacy_sequence]

        objective_names = self._extract_objective_name_chunks()

        for phrase in objective_names:

            if any(char in special_characters for char in phrase):
                match_string = r'{}'.format(re.escape(phrase.lower()))
            else:
                match_string = r'\b{}\b'.format(phrase.lower())

            match_result = re.finditer(match_string, self._spacy_sequence.text.lower())

            for m in match_result:

                start = self._word_dict[m.start()][0]
                end = start + len(m.group().split())

                for idx in range(start, end):

                    if spans[idx] == 'O':
                        if idx == start:
                            spans[idx] = "B-OBJ_NAME"
                        else:
                            spans[idx] = "I-OBJ_NAME"

        return spans

    def _get_noun_chunk_conjuncts(self, noun_chunks_list: List[SpanType]) -> List[str]:
        """Get all conjuncting noun chunks."""

        conjuncting_noun_chunks = []

        for noun_chunk_index, noun_chunk in enumerate(noun_chunks_list):
            # Often spacy noun conjuncts doesn't work on the entire phrase
            # so multiple cases are to be explored to find the conjuncts

            # noun conjuncts
            case_one = noun_chunk.conjuncts
            # the previous token conjuncts
            case_two = self._spacy_sequence[noun_chunk.start - 1].conjuncts
            # first word of the chunk conjuncts
            case_three = noun_chunk[0].conjuncts

            if case_one or case_two or case_three:

                # extract the conjuncting phrase here
                phrase = noun_chunk

                if not noun_chunk.conjuncts:

                    if noun_chunk[0].conjuncts:
                        phrase = noun_chunk[0]
                        conjuncting_phrase = noun_chunk[0].conjuncts[0]
                    else:
                        if not self._spacy_sequence[noun_chunk.start - 1].conjuncts[0].i < noun_chunk.start:
                            conjuncting_phrase = noun_chunks_list[noun_chunk_index - 1]
                        else:
                            if noun_chunk_index + 1 < len(noun_chunks_list):
                                conjuncting_phrase = noun_chunks_list[noun_chunk_index + 1]

                else:

                    if noun_chunk.conjuncts[0].i < noun_chunk.start:
                        conjuncting_phrase = noun_chunks_list[noun_chunk_index - 1]
                    else:
                        conjuncting_phrase = noun_chunks_list[noun_chunk_index + 1]

                conjuncting_noun_chunks += self._filter_noun_chunk_conjuncts([phrase, conjuncting_phrase])

        return conjuncting_noun_chunks

    def _get_conjuncting_chunks(self) -> List[str]:
        """Extract all conjuncting chunks/key-phrases."""

        conjuncting_chunks = []

        noun_chunks_list = list(self._spacy_sequence.noun_chunks)

        # get conjuncting noun chunks
        conjuncting_noun_chunks = self._get_noun_chunk_conjuncts(noun_chunks_list)
        conjuncting_chunks += conjuncting_noun_chunks

        # add all phrases connected via hyphens and quotes
        hyphens_and_quotes = [item.text for item in get_phrases_with_hyphens_and_quotes(self._spacy_sequence)]
        conjuncting_chunks += hyphens_and_quotes

        # add all other missing noun chunks which start with conjuncting items
        missing_chunks = []
        tmp_chunks = []

        for val in noun_chunks_list:
            if val[0].pos_ == 'DET':
                tmp_chunks.append(val[1::].text)
            else:
                tmp_chunks.append(val.text)

        for chunk in tmp_chunks:
            lowered_chunk = chunk.lower()
            for item in conjuncting_chunks:
                if lowered_chunk.startswith(item.lower()):
                    missing_chunks.append(lowered_chunk)

        conjuncting_chunks += missing_chunks
        conjuncting_chunks = list(set(conjuncting_chunks))

        # add all prepositional chunks which end with conjuncting items
        prepositional_subtrees = self._extract_prepositional_chunks(condition_on_obj_dir_keywords=False)
        prepositional_conjuncts = self._get_prepositional_chunk_conjuncts(prepositional_subtrees, conjuncting_chunks)
        conjuncting_chunks += prepositional_conjuncts

        # add all singulars and plurals
        singulars_and_plurals = []

        for val in conjuncting_chunks:
            singulars_and_plurals += generate_singulars_and_plurals([val])

        conjuncting_chunks = list(set(conjuncting_chunks + singulars_and_plurals))

        return conjuncting_chunks

    def get_variable_spans(self) -> List[str]:
        """Identify variable spans in a given sequence."""

        spans = ['O' for _ in self._spacy_sequence]

        conjuncting_chunks = self._get_conjuncting_chunks()
        conjuncting_chunks.sort(key=lambda x: len(x.split()), reverse=True)

        for phrase in conjuncting_chunks:

            if any(char in special_characters for char in phrase):
                match_string = r'{}'.format(re.escape(phrase.lower()))
            else:
                match_string = r'\b{}\b'.format(phrase.lower())

            match_result = re.finditer(match_string, self._spacy_sequence.text.lower())

            for m in match_result:

                try:
                    start = self._word_dict[m.start()][0]
                except KeyError:
                    continue

                end = start + len(m.group().split())

                for idx in range(start, end):

                    if spans[idx] == 'O':
                        if idx == start:
                            spans[idx] = "B-VAR"
                        else:
                            spans[idx] = "I-VAR"
        return spans

    def get_prepositional_chunk_spans(self) -> List[str]:
        """Identify preposition chunk spans in a given sequence."""

        prepositional_chunk_spans = ['O' for _ in self._spacy_sequence]
        prepositional_subtrees = self._extract_prepositional_chunks(condition_on_obj_dir_keywords=False)

        for subtree in prepositional_subtrees:

            start = subtree.start
            end = start + len(subtree.text.split())

            for idx in range(start, end):

                if idx == start:
                    prepositional_chunk_spans[idx] = "B-PREP_CHUNK"
                else:
                    prepositional_chunk_spans[idx] = "I-PREP_CHUNK"

        return prepositional_chunk_spans

    def get_linguistic_features(self) -> List[Tuple]:
        """Extract all linguistic features in a given sequence."""

        noun_chunk_labels = ['O' for _ in self._spacy_sequence]

        # Noun Chunks
        for chunk in self._spacy_sequence.noun_chunks:
            for chunk_idx in range(chunk.start, chunk.end):
                if noun_chunk_labels[chunk_idx] == 'O':
                    if chunk_idx == chunk.start:
                        noun_chunk_labels[chunk_idx] = "B-NOUN_CHUNK"
                    else:
                        noun_chunk_labels[chunk_idx] = "I-NOUN_CHUNK"

        # Gold Entities
        entity_tags = self._mark_entities()

        # All the other features
        features = []

        for token_index, token in enumerate(self._spacy_sequence):

            lower_case_token = token.text.lower()
            in_nltk_words = lower_case_token in word_set
            in_nltk_people_names = token.text in name_set

            features.append(
                (token.text, lower_case_token, token.text.istitle(), token.text.isupper(), token.lemma_, token.pos_,
                 token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, in_nltk_words,
                 in_nltk_people_names, noun_chunk_labels[token_index], entity_tags[token_index]))

        return features

    def get_features(self) -> List[Tuple]:
        """Extract all features relevant for a given model type."""

        linguistic_features = self.get_linguistic_features()
        prepositional_chunks = self.get_prepositional_chunk_spans()

        features = []

        for token_index, token_features in enumerate(linguistic_features):
            tmp_features = token_features + (prepositional_chunks[token_index],)
            features.append(tmp_features)

        if self._model_name in [Constants.LEXICAL_PLUS, Constants.HYBRID]:

            # These features common for both lexical_plus and hybrid

            # Gazetteer Features
            obj_dir_gazetteer_features = self.get_gazetteer_features(Constants.OBJ_DIR_KEYWORDS, "OBJ_DIR")
            const_dir_gazetteer_features = self.get_gazetteer_features(Constants.CONST_DIR_KEYWORDS, "CONST_DIR")
            # Objective name automaton features
            obj_name_features = self.get_objective_name_spans()
            # Var name features
            var_name_features = self.get_variable_spans()

            # Concatenate features
            for token_index, token_features in enumerate(features):
                tmp_features = (token_features + (
                    obj_dir_gazetteer_features[token_index],
                    const_dir_gazetteer_features[token_index],
                    obj_name_features[token_index],
                    var_name_features[token_index],
                ))
                features[token_index] = tmp_features

        return features
