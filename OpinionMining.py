#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from pandas import read_excel
from collections import Counter
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from stop_words import get_stop_words
from spacy.lang.es import Spanish

def read_as_list(l, encoding):
    l_ = []
    with open(l, "rt", encoding=encoding) as f:
        l_ = f.read().splitlines()
    return l_

def filter_vocabulary(l, pcg):
    vocab, words, l_filtered = [], [], []
    for l_ in l:
        for l__ in l_.split():
            vocab.append(l__)

    c = Counter(vocab)
    c = c.most_common()
    n = int(np.floor(pcg * len(c)))

    for w in c:
        words.append(w[0])

    avoid = words[0:n] + words[-n:-1]

    for l_ in l:
        l_filtered.append(' '.join([x for x in l_.split() if x not in avoid]))

    return l_filtered

def tokenize(text, parser):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)

    return lda_tokens

def prepare_text_for_ML(x, stop_words, parser, stemmer, to_avoid):
    tokens = tokenize(x, parser)
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop-words.
    tokens = [x for x in tokens if not isinstance(x, int)]  # Remove integers.
    tokens = [x for x in tokens if len(x) > 3]  # Remove words with less than 3 letters.
    tokens = [stemmer.stem(token) for token in tokens]  # Lemmatize words.
    tokens = [x for x in tokens if x not in to_avoid]

    return ' '.join(tokens)

if '__main__' == __name__:

    stemmer = SnowballStemmer('spanish')
    sys.setrecursionlimit(10000)

    cwd = os.getcwd()

    stop_words = get_stop_words('es')
    parser = Spanish()

    to_avoid = read_as_list('to_avoid.txt', 'latin-1')

    my_sheet = 'Sheet1'
    file_name = 'Proposals - PAM - Spanish.xlsx'  # name of your excel file
    df = read_excel(file_name, sheet_name=my_sheet)
    df = df[df['category/name/se'] == 'Sanidad y salud']

    txt = list(df['body'])

    text = [filter_vocabulary(txt, 0.01)][0]
    text = [prepare_text_for_ML(x, stop_words, parser, stemmer, to_avoid) for x in text]

    count_vect = CountVectorizer(max_df=0.8, min_df=2)
    doc_term_matrix = count_vect.fit_transform(text)

    LDA = LatentDirichletAllocation(n_components=5, random_state=42)
    LDA.fit(doc_term_matrix)

    for i, topic in enumerate(LDA.components_):
        print(f'Topic #{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
