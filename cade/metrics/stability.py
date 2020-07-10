from sklearn.metrics.pairwise import paired_cosine_distances


def shared_voc(mod_a, mod_b):
    """
    Return set of shared vocabulary elements
    """
    voc_a = set(mod_a.wv.vocab.keys())
    voc_b = set(mod_b.wv.vocab.keys())
    return voc_a.intersection(voc_b)


def jumpers(mod_a, mod_b, top_n=20):
    """
    Return <top_n> vocabulary elements with lower cosine similarity between models
    """

    vocab = shared_voc(mod_a, mod_b)
    sims = 1 - paired_cosine_distances(mod_a.wv[vocab], mod_b.wv[vocab])
    sim_dict = {k: v for k, v in zip(vocab, sims)}

    return sorted(sim_dict, key=sim_dict.get)[:top_n]


def stables(mod_a, mod_b, top_n=20):
    """
    Return <top_n> vocabulary elements with lower cosine similarity between models
    """

    vocab = shared_voc(mod_a, mod_b)
    sims = 1 - paired_cosine_distances(mod_a.wv[vocab], mod_b.wv[vocab])
    sim_dict = {k: v for k, v in zip(vocab, sims)}

    return sorted(sim_dict, key=sim_dict.get, reverse=True)[:top_n]
