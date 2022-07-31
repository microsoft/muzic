"""Use Textrank and TF-IDF to calculate keyword
"""
import os
import pickle
import joblib
from textrank4zh import TextRank4Keyword

def dump_file(filename, obj) -> None:
    """Using Pickle to dump object

    Args:
        filename (string): filename
        obj (any): obj to be dumped
    """
    if os.path.isfile(filename):
        raise FileExistsError

    with open(filename, "wb") as o:
        pickle.dump(obj, o)

def load_file(filename):
    """Using pickle to load object back

    Args:
        filename (string): filename
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError

    with open(filename, "rb") as f:
        obj = pickle.load(f)

    return obj

def get_stop_words(path) -> list:
    with open(path, "r") as stop_file:
        stop_words = stop_file.readlines()

    stop_words = [ s.strip("\n") for s in stop_words ]
    return stop_words

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

def infer_tfidf(text: list, model) ->dict:
    # load model
    tf = joblib.load(model)

    feature_names = tf.get_feature_names()

    # infer
    result_vector = tf.transform([text])
    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(result_vector.tocoo())
    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    return keywords

def get_textrank(text, topk=10) -> list:
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)

    rank = [ (item.word, item.weight) for item in tr4w.get_keywords(topk, word_min_len=1) ]

    return rank

def get_keyword(text) -> dict:
    """Using TF-IDF and TextRank to find keywords

    Args:
        text (string): raw text

    Returns:
        results (dict): {[keywords]: [score]}
    """
    a = 0.5
    textrank = dict(get_textrank(text), topk=5)
    # textrank = dict(get_textrank(text))
    # tf_idf = infer_tfidf(text, "lyrics_tfidf_model.pkl")

    # keys = list(textrank.keys())
    # keys = list(textrank.keys() & tf_idf.keys())
    # tmp = {}
    # for key in keys:
    #     textr_score = textrank[key]
    #     tfidf_score = tf_idf[key]

    #     tmp[key] = textr_score * (1-a) + tfidf_score * a

    # keys = sorted(tmp, key=tmp.get)
    
    keys = sorted(textrank, key=textrank.get)
    # results = { k: tmp[k] for k in keys }
    results = { k: textrank[k] for k in keys }
    return results