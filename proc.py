"""
processing
"""

def count_tokens(path):
	"""
	helper/count_tok.py
	"""
	return count


def most_frequent_nouns(path):
	"""
	met-cluster/extract.py + helper/wordsplitter_dict.py
	"""
	return mfn


def count_features(nouns, feats):
	"""
	met-cluster/features.py
	"""
	return matrix, nouns_dict, feats_dict


def scale_features(matrix):
	"""
	met-cluster/features.py + met-cluster/cluster.py
	"""
	return matrix


def filter_min_freq(nouns, feats):
	"""
	met-cluster/features.py
	"""
	return nouns, feats


def filter_top_feat(nouns, feats):
	"""
	met-cluster/features.py
	"""
	return nouns, feats


def filter_by_mfn(nouns, feats):
	"""
	met-cluster/features.py
	"""
	return nouns, feats


def filter_by_pos(nouns, feats):
	"""
	met-cluster/features.py
	"""
	return nouns, feats


def jsd(vec1, vec2):
	"""
	met-cluster/features.py
	"""
	return distance


def similarity_matrix(matrix):
	"""
	met-cluster/features.py
	"""
	return matrix


def distance_to_similarity(df):
	"""
	met-cluster/cluster.py
	"""
	return sim


def lda(path, save_path):
	"""
	topic modeling/lda.py
	"""


def lsa(path, save_path):
	"""
	word embedding/lsa.py
	"""


def doc2vec(path, save_path):
	"""
	word embedding/doc2vec.py
	"""


def clf(matrix, save_path):
	"""
	van halteren/clf_alltokens.py
	"""


def select_features_vh(path):
	"""
	van halteren/vh_tutorial.py
	"""
	return matrix


def tfidf_calculate(path):
	"""
	met-sampler/classify.py
	"""
	return df, clf


def tfidf_metaphor_clf(clf, df, str):
	"""
	met-sampler/classify.py
	"""
	return boolean
