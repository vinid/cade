import itertools
import math

import numpy as np
from scipy.spatial.distance import cosine

avg_m1 = 0
avg_m2 = 0
initialized_avgs = False


def initialize_avgs(m1, m2):
    global avg_m1, avg_m2, initialized_avgs
    avg_m1 = np.average(m1[m1.wv.vocab], axis=0)
    avg_m2 = np.average(m2[m2.wv.vocab], axis=0)
    initialized_avgs = True


def get_neighbors_set(word, slice_year, topn):
    return set([k[0] for k in slice_year.wv.most_similar(word, topn=topn)])


def c_measure(word, slices):
    word_vectors = [slice[word] for slice in slices]

    combs = itertools.combinations(word_vectors, 2)

    collect_c = []
    for a, b in combs:
        collect_c.append(1 - cosine(a, b))
    return collect_c


def lncs2_setted(word, m1, m2, topn):
    """
    https://www.aclweb.org/anthology/D16-1229/

    :param word:
    :param m1:
    :param m2:
    :param topn:
    :return:
    """

    global avg_m1, avg_m2, initialized_avgs

    words_m1 = list(get_neighbors_set(word, m1, topn))
    words_m2 = list(get_neighbors_set(word, m2, topn))

    all_words = set(words_m1 + words_m2)

    vec_1 = []
    vec_2 = []
    mean = False or initialized_avgs

    for inner_word in all_words:
        if inner_word in m1.wv.vocab:
            vec_1.append(1 - cosine(m1.wv[word], m1.wv[inner_word]))
        else:
            if not mean:
                avg_m1 = np.average(m1[m1.wv.vocab], axis=0)
                mean = True
            vec_1.append(1 - cosine(m1.wv[word], avg_m1))

    mean = False or initialized_avgs

    for inner_word in all_words:
        if inner_word in m2.wv.vocab:
            vec_2.append(1 - cosine(m2.wv[word], m2.wv[inner_word]))
        else:
            if not mean:
                avg_m2 = np.average(m2[m2.wv.vocab], axis=0)
                mean = True
            vec_2.append(1 - cosine(m2.wv[word], avg_m2))

    return 1 - cosine(vec_1, vec_2)


def lncs2(word, m1, m2, topn):
    """
    https://www.aclweb.org/anthology/D16-1229/

    :param word:
    :param m1:
    :param m2:
    :param topn:
    :return:
    """

    global avg_m1, avg_m2, initialized_avgs

    words_m1 = list(get_neighbors_set(word, m1, topn))
    words_m2 = list(get_neighbors_set(word, m2, topn))

    vec_1 = []
    vec_2 = []
    mean = False or initialized_avgs

    # Cosine similarity between "word" and every word in its m1-neighbour
    # within the m1 space
    for wtest in words_m1:
        vec_1.append(1 - cosine(m1.wv[word], m1.wv[wtest]))

    # Cosine similarity between "word" and every word in its m2-neighbour
    # within the m1 space
    for wtest in words_m2:
        if wtest not in m1.wv.vocab:
            if not mean:
                # Represent OOV words in m1 space empirically with its mean
                avg_m1 = np.average(m1[m1.wv.vocab], axis=0)
                mean = True
            vec_1.append(1 - cosine(m1[word], avg_m1))
        else:
            vec_1.append(1 - cosine(m1.wv[word], m1.wv[wtest]))

    mean = False or initialized_avgs

    # Cosine similarity between "word" and every word in its m1-neighbour
    # within the m2 space
    for wtest in words_m1:
        if wtest not in m2.wv.vocab:
            if not mean:
                # Represent OOV words in m1 space empirically with its mean
                avg_m2 = np.average(m2[m2.wv.vocab], axis=0)
                mean = True
            vec_2.append(1 - cosine(m2[word], avg_m2))
        else:
            vec_2.append(1 - cosine(m2.wv[word], m2.wv[wtest]))

    # Cosine similarity between "word" and every word in its m2-neighbour
    # within the m2 space
    for wtest in words_m2:
        vec_2.append(1 - cosine(m2.wv[word], m2.wv[wtest]))

    return 1 - cosine(vec_1, vec_2)


def get_mean_if_missing(word1, word2, m1, m2, mean=False):
    """

    :param word1:
    :param word2:
    :param m1:
    :param m2:
    :param mean:
    :return:
    """
    if mean:
        avg = np.average(m1[m1.wv.vocab], axis=0)
        return 1 - cosine(m1.wv[word1], avg)
    else:
        return 1 - cosine(m1.wv[word1], m2.wv[word2])


def moving_lncs2(word, m1, m2, topn, t):
    if math.isclose(0.0, t):
        return lncs2(word, m1, m2, topn)
    elif math.isclose(1.0, t):
        return 1 - cosine(m1[word], m2[word])
    else:
        return (1 - t) * lncs2(word, m1, m2, topn) + t * (
            1 - cosine(m1[word], m2[word])
        )


def intersection_nn(word, m1, m2, topn=1000):
    """
    https://www.aclweb.org/anthology/2020.acl-main.51/

    :param word:
    :param m1:
    :param m2:
    :param topn:
    :return:
    """

    assert topn > 0
    words_m1 = get_neighbors_set(word, m1, topn)
    words_m2 = get_neighbors_set(word, m2, topn)
    intersection = words_m1.intersection(words_m2)
    return len(intersection) / topn
