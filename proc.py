"""
processing
"""

import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
from os.path import basename
from joblib import Parallel, delayed, load, dump
from scipy.spatial.distance import cosine, euclidean, cityblock, jaccard
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, Word2Vec, FastText
from io import save_pkl, load_pkl, strip_symbols


def token_count(path, pos=True, pos_exclude="PUNC", pos_column="CPOS"):
	"""
	return corpus size in tokens

		- example:
		
			>>> size = count_tokens("~/Daten/romankorpus")

	@param path: path to directory containing CoNLL2009/DOF files (tab-delimited)
	@param pos_filter: filter by POS tag (default: True)
	@param pos_exclude: POS tag to excude (default: "PUNC")
	@param pos_column: name of POS column (default: "CPOS")
	@return: int
	"""
	count = 0

	for filepath in glob(path):
		if not filepath.startswith('.'):
			try:
				df = pd.read_csv(filepath, sep="\t")
			except (pd.parser.CParserError) as detail:
				print(filepath, detail)

			if pos_filter is True:
				pos = df.groupby(pos_column)
				df = df.drop(pos.get_group(pos_exclude).index)       # don't count the punctuation

			count += len(df.index)

	return count


def most_frequent(path, top_n, type=True, pos=True, pos_tag="NN", pos_column="CPOS"):
	"""
	count most frequent tokens or types
	
		- example:
		
			>>> mfn_dict = count_most_frequent("~/Daten/romankorpus", 50000, pos_tag="NN")
			>>> tokens_dict = count_most_frequent("~/Daten/romankorpus", 150000, type=False, pos=False)

	@param path: path to directory containing CoNLL2009/DOF files (tab-delimited)
	@param top_n: number of most frequent items to return
	@param type: count types only (expects "Lemma" column, default: True)
	@param pos: count pos_tag only (default: True)
	@param pos_tag: target POS (default: "NN")
	@param pos_column: name of POS column (default: "CPOS")
	@return: dict
	"""
	tokens = []

	for filepath in glob(path):
		if not filepath.startswith('.'):
			try:
				df = pd.read_csv(filepath, sep="\t", quoting=csv.QUOTE_NONE)
			except (pd.parser.CParserError) as detail:
				print(filepath, detail)

			if pos_filter is True:
				df_grouped = df.groupby(pos_column)
				df = df_grouped.get_group(pos_tag)

			if type_filter is True:
				for token in df['Lemma'].tolist():
					if '|' in token: token = token.split('|')[0]   # treetagger format
					tokens.append(strip_symbols(token.lower()))
			else:
				tokens += [strip_symbols(token.lower()) for token in df['Token'].tolist()]

	tokens = filter(None, tokens)     # strip_symbols might have produced empty list items, filter them out
	
	c = Counter(tokens)
	if len(c) < max_count: max_count = len(c)

	return dict(c.most_common(max_count))


def sim2dist(X):
	"""
	conversion between similarity and distance matrix

	goes both ways, expects distances between 0-1

	pandas version: df.apply(lambda x: 1-x, raw=True)
	"""
	return X.vectorize(lambda x: 1-x)


def jsd(x, y):
	"""
	Jensen-Shannon divergence

	@param x, y: vectors as list or array
	@return: float
	"""
	x = np.array(x)
	y = np.array(y)
	d1 = x*np.log2(2*x/(x+y))
	d2 = y*np.log2(2*y/(x+y))
	d1[np.isnan(d1)] = 0
	d2[np.isnan(d2)] = 0
	d = 0.5*np.sum(d1+d2)

	return d


class FrequencyMatrix:
	"""
	build frequency matrix

		- example:
		
			>>> freq = FrequencyMatrix(words, feats)
			>>> freq.prune_minfeat(800)
			>>> freq.build()
			>>> freq.norm()
			>>> X = freq.X
	"""
	def __init__(words, feats):
		"""
		initialize

		@param words: list of strings - one word for each feature in the feats list
		@param feats: list of strings - one feature for each word in the words list
		"""
		self.words = words
		self.feats = feats
		self.words_orig = words  # preserve original data for reset()
		self.feats_orig = feats
		self.X = None
		self.words_dict = {}
		self.feats_dict = {}
		self.info = []  # take note of applied transformations (pruning, norm)

	def build(self):
		"""
		construct matrix
		"""
		words = self.words
		feats = self.feats

		words_unique = sorted(list(set(words)))                                 # column header
		feats_unique = sorted(list(set(feats)))                                 # index

		words_dict = {word: i for i, word in enumerate(words_unique)}           # build word-id dictionary
		words_ids = [words_dict[word] for word in words]                        # translate words into ids

		feats_dict = {feat: i for i, feat in enumerate(feats_unique)}           # same for feats
		feats_ids = [feats_dict[feat] for feat in feats]                        # cf. http://stackoverflow.com/a/17152507

		X = np.zeros(shape=(len(feats_unique), len(words_unique)), dtype=np.int64)

		for n_id, f_id in zip(words_ids, feats_ids):
			if X[f_id, n_id] == 0.0:
				X[f_id, n_id] = 1
			else:
				X[f_id, n_id] += 1

		self.X = X
		self.words_dict = words_dict
		self.feats_dict = feats_dict

	def norm(self):
		"""
		L1 normalise frequency matrix
		"""
		X = self.X
		X_scale = np.zeros(shape=(X.shape[0], X.shape[1]), dtype=np.float64)

		for index1 in range(X.shape[1]):
			X_scale[:,index1] = X[:,index1] / X[:,index1].sum()

		self.info.append("L1 norm")
		self.X = X_scale

	def prune_by_freq(self, min_freq, inverse=False):
		"""
		use only words with a frequency higher than min_freq

		@param min_feat: minimum frequency for each word
		@param inverse: use all except the most common (default: False)
		"""
		c = Counter(self.words)
		if inverse == False:
			words_selected = [w for w, count in c if count >= min_freq]
		else:
			words_selected = [w for w, count in c if count <= min_freq]
		
		words_filtered = []
		feats_filtered = []
		for w, f in zip(self.words, self.feats):
			if w in words_selected:
				words_filtered.append(w)
				feats_filtered.append(f)

		self.info.append("prune_by_freq="+",".join(labels)+"inverse="+str(inverse))
		self.words = words_filtered
		self.feats = feats_filtered

	def prune_by_featlabel(self, labels, inverse=False):
		"""
		use only features containing certain substrings

		@param labels: list of label strings, e.g. ['-v', '-adj', '-pp', '-gen', '-comp']
		@param inverse: exclude features with those labels (default: False)
		"""
		words_filtered = []
		feats_filtered = []

		for w, f in zip(self.words, self.feats):
			if w != None and f != None:
				if inverse == False:
					for l in use_labels:
						if l in f:
							words_filtered.append(w)
							feats_filtered.append(f)
				else:
					if not any(l in f for l in use_labels):
						words_filtered.append(w)
						feats_filtered.append(f)

		self.info.append("prune_by_featlabel="+",".join(labels)+"inverse="+str(inverse))
		self.words = words_filtered
		self.feats = feats_filtered

	def prune_by_featfreq(self, min_feat):
		"""
		use only words with more than min_feat unique features

		@param min_feat: minimum number of features for each word
		"""
		gluedlist = []
		for w, f in zip(self.words, self.feats):
			if w is not None and f is not None:
				gluedlist.append(str(w)+'###'+str(f))

		# drop duplicates
		gluedlist = list(set(gluedlist))

		# splitting data
		words_unique = []
		feats_unique = []
		for wf in gluedlist:
			w, f = wf.split('###')
			words_unique.append(w)
			feats_unique.append(f)

		# select words where count > min_feat
		words_selected = []
		c = Counter(words_unique)
		for k in list(c):
			if c[k] >= min_feat:
				words_selected.append(k)

		# filter words+feats using the words_selected list
		words_filtered = []
		feats_filtered = []
		for w, f in zip(self.words, self.feats):
			if w in words_selected and w != None and f != None:
				words_filtered.append(w)
				feats_filtered.append(f)

		self.info.append("prune_by_featfreq="+str(min_feat))
		self.words = words_filtered
		self.feats = feats_filtered

	def prune_by_featfreq2(self, top_feat, inverse=False):
		"""
		use only the most common features overall

		@param top_feat: number of most common features
		@param inverse: use all except the most common (default: False)
		"""
		c = Counter(self.feats)
		feats_selected = [f for f, count in c.most_common(top_feat)]
		
		words_filtered = []
		feats_filtered = []
		for w, f in zip(self.words, self.feats):
			if inverse == False:
				if f in feats_selected:
					words_filtered.append(w)
					feats_filtered.append(f)
			else:
				if f not in feats_selected:
					words_filtered.append(w)
					feats_filtered.append(f)

		self.info.append("prune_by_featfreq2="+str(top_feat)+"inverse="+str(inverse))
		self.words = words_filtered
		self.feats = feats_filtered

	def reset(self):
		"""
		reset filters
		"""
		self.words = self.words_orig
		self.feats = self.feats_orig
		self.info = []

	def save(self, path="FrequencyMatrix.pkl"):
		"""
		save class to disk

		@param path: file to save (default: "FrequencyMatrix.pkl")
		"""
		save_pkl(path, self)

	def load(self, path="FrequencyMatrix.pkl"):
		"""
		load class from disk

		@param path: file to load (default: "FrequencyMatrix.pkl")
		"""
		self = load_pkl(path)


class SimilarityMatrix:
	"""
	build similarity matrix

		- example:
		
			>>> sim = SimilarityMatrix(X)
			>>> sim.build("jsd")
			>>> W = sim.W
			>>> dist = sim.to_distance()
	"""
	def __init__(X):
		"""
		initialize

		@param X: frequency matrix
		"""
		self.X = X
		self.W = None
		self.info = []  # take note of distance metric used

	def build(self, metric, n_workers=4):
		"""
		build similarity matrix, compute weights w/ joblib.parallel

		@param metric: distance metric ("jsd", "cosine", "euclidean", "manhattan", "jaccard")
		@param n_workers: number of parallel processes (default: 4)
		"""
		if metric is "jaccard":
			X = self.X > 0  # comparison results in boolean array
		else:
			X = self.X
		indices = np.array_split(range(X.shape[1]), n_workers)

		#now = time.time()

		# max_nbytes threshold triggers automatic memmapping of input data
		# cf. https://pythonhosted.org/joblib/parallel.html#working-with-numerical-data-in-shared-memory-memmaping
		#res = Parallel(n_jobs=n_workers)(delayed(_calc_weight) (X, ind, metric) for ind in indices)
		res = Parallel(n_jobs=n_workers, max_nbytes=1e6)(delayed(self._calc_weight) (X, ind, metric) for ind in indices)
		
		#print('Parallel, n_jobs=', n_workers, ': finished in', time.time()-now , 'sec\n')

		result = []
		for r in res: result += r

		W = np.zeros(shape=(X.shape[1], X.shape[1]), dtype=np.float64)

		for index1, index2, weight in result:
			W[index1, index2] = weight

		self.info.append(metric)
		self.W = W

	def _calc_weight(X, indices_slice, metric):
		"""
		calculate weight using distance metric and return 3-tuples
		"""
		indices_all = range(X.shape[1])       # splitted array part has new index from 0..n
		result = []

		for index1 in indices_slice:
			for index2 in indices_all:
				if metric == 'jsd':
					weight = jsd(X[:,index1], X[:,index2])
				elif metric == 'cosine':
					weight = cosine(X[:,index1], X[:,index2])
				elif metric == 'euclidean':
					weight = euclidean(X[:,index1], X[:,index2])
				elif metric == 'manhattan':
					weight = cityblock(X[:,index1], X[:,index2])
				elif metric == 'jaccard':
					weight = jaccard(X[:,index1], X[:,index2])  # comparison returns boolean index
				"""
				elif metric == 'mh':
					wmg = WeightedMinHashGenerator(len(indices_all))
					wm1 = wmg.minhash(X[:,index1])          # cf. https://github.com/ekzhu/datasketch#weighted-minhash
					wm2 = wmg.minhash(X[:,index2])
					weight = wm1.jaccard(wm2)
				"""
				result.append((index1, index2, weight))     # save weight together with indices for the complete matrix

		return result

	def save(self, path="SimilarityMatrix.pkl"):
		"""
		save class to disk

		@param path: file to save (default: "SimilarityMatrix.pkl")
		"""
		save_pkl(path, self)

	def load(self, path="SimilarityMatrix.pkl"):
		"""
		load class from disk

		@param path: file to load (default: "SimilarityMatrix.pkl")
		"""
		self = load_pkl(path)

"""
TODO:

- semanticmodel

	- preprocessing -> s.u.

	- neu: word2vec, fasttext

	- metadata management
	   - als doc_labels vorhanden (= filename + doc id)
	   - andere quellen: dataframe/excel tabellen ?
	   - datatype: dict ?
	      -> vgl. funktionen fuer uwue-korpora: selben datatype verwenden
"""

class SemanticModel:
	"""
	model a corpus using lda, lsa, word2vec, or fasttext

		- example:
		
			>>> sem = SemanticModel("~/Daten/romankorpus")
			>>> sem.preproc(doc_split=True, pos_filter=True)
			>>> sem.lda(model_parameters)
			>>> sem.save("romankorpus_lda")
	"""
	def __init__(corpus_path):
		self.corpus_path = corpus_path
		self.doc_split = None
		self.doc_size = None
		self.stopword_filter = None
		self.stopword_path = None
		self.pos_filter = None
		self.pos_tags = None
		self.pos_column = None
		self.bow_dictionary = None
		self.bow_corpus = None
		self.doc_labels = None  # metadata
		self.info = []  # take note of model parameters

	def preproc(self, doc_split=False, doc_size=1000, stopword_filter=False, stopword_path="stopwords.txt",
		pos_filter=False, pos_tags=["ADJ", "NN", "V"], pos_column="CPOS"):
		"""
		preprocessing
		"""
		self.doc_split = doc_split
		self.doc_size = doc_size
		self.stopword_filter = stopword_filter
		self.stopword_path = stopword_path
		self.pos_filter = pos_filter
		self.pos_tags = pos_tags
		self.pos_column = pos_column

		path = self.corpus_path
		docs = []
		doc_labels = []
		stopwords = []

		if stopword_filter is True:
			with open(stopwordlist, 'r') as f: stopwords = f.read()
			stopwords = sorted(set(stopwords.split("\n")))

		# TODO: das folgende ausgliedern in neue funktion?
		# for p in paths -> yield doc

		paths = [p for p in glob(path) if not p.startswith(".")]
		for p in paths:

			if p.endswith(".txt"):
				with open(file, "r") as f:
					if doc_split is True:
						docs.append(strip_symbols(f.read(doc_size)))        # TODO: read() expects size in bytes, we have words
						doc_labels.append(basename(p))                      # numerate f.read chunks
					else:
						docs.append(strip_symbols(f.read()))
						doc_labels.append(basename(p))                      # metadata, e.g. used as plot labels
					f.close()

			elif p.endswith(".csv") or p.endswith(".tsv"):
				df = pd.read_csv(filepath, sep="\t", quoting=csv.QUOTE_NONE)
	
				if pos_filter is True:
					df = df.groupby(pos_column)
					doc = pd.DataFrame()
					for t in pos_tags:
						doc = doc.append(df.get_group(t))

					#names = df.get_group('NP')['Lemma'].values.astype(str)
					#stopwords += names.tolist()
				else:
					doc = df

				# construct documents
				if doc_split is True:
					doc = doc.sort(columns='TokenId')
					i = 1
					while(doc_size < doc.shape[0]):
						docs.append(doc[:doc_size]['Lemma'].values.astype(str))
						doc_labels.append(file.split(".")[0]+" #"+str(i))   # metadata, e.g. used as plot labels
						doc = doc.drop(doc.index[:doc_size])        # drop doc_size rows
						i += 1
					docs.append(doc['Lemma'].values.astype(str))    # add the rest
					doc_labels.append(file.split(".")[0]+" #"+str(i))

				docs = [[strip_symbols(word) for word in doc] for doc in docs]


		texts = docs

		# remove stopwords
		texts = [[word for word in doc if word not in stopwords] for doc in docs]

		# TODO: filter extremes -> add to stopwords
		# vgl. https://tedboy.github.io/nlps/generated/generated/gensim.corpora.Dictionary.filter_extremes.html
		
		#for doc in docs:
			#print(str(len(doc)))              # display resulting doc sizes


		# remove words that appear only once
		all_tokens = sum(texts, [])
		tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
		texts = [[word for word in text if word not in tokens_once] for text in texts]

		# vectorize
		dictionary = Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]

		self.bow_dictionary = dictionary
		self.bow_corpus = corpus
		self.doc_labels = doc_labels

	def lda(self, model_parameters):
		"""
		topic modeling/lda.py
		"""

	def lsa(self, model_parameters):
		"""
		word embedding/lsa.py
		"""

	def word2vec(self, model_parameters):
		"""
		neu
		"""

	def fasttext(self, model_parameters):
		"""
		neu
		"""

	def save(self, path="SemanticModel.pkl"):
		"""
		save class to disk

		@param path: file to save (default: "SemanticModel.pkl")
		"""
		save_pkl(path, self)

	def load(self, path="SemanticModel.pkl"):
		"""
		load class from disk

		@param path: file to load (default: "SemanticModel.pkl")
		"""
		self = load_pkl(path)


class Classifiers:

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
