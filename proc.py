"""
processing
"""

import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
from os.path import basename
from datetime import datetime
from inspect import currentframe, getargvalues
from joblib import Parallel, delayed, load, dump
from scipy.spatial.distance import cosine, euclidean, cityblock, jaccard
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, Word2Vec, FastText
from io import save_pkl, load_pkl, strip_symbols


def token_count(path, pos_filter=True, pos_exclude="PUNC", pos_column="CPOS"):
	"""
	return corpus size in tokens

		- example:
		
			>>> size = token_count("~/Daten/romankorpus")

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


def most_frequent(path, top_n, type_filter=True, pos_filter=True, pos_tag="NN", pos_column="CPOS"):
	"""
	count most frequent tokens or types
	
		- example:
		
			>>> mfn_dict = most_frequent("~/Daten/romankorpus", 50000, pos_tag="NN")
			>>> tokens_dict = most_frequent("~/Daten/romankorpus", 150000, type=False, pos=False)

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
	if len(c) < top_n: top_n = len(c)

	return dict(c.most_common(top_n))


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
			>>> freq.prune_by_featfreq(800)
			>>> freq.build()
			>>> freq.norm()
			>>> X = freq.X
	"""
	def __init__(self, words, feats):
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

		self.info.append("prune_by_freq="+str(min_freq)+", inverse="+str(inverse))
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
					for l in labels:
						if l in f:
							words_filtered.append(w)
							feats_filtered.append(f)
				else:
					if not any(l in f for l in labels):
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
		path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+path
		save_pkl(path, self.__dict__)

	def load(self, path="FrequencyMatrix.pkl"):
		"""
		load class from disk

		@param path: file to load (default: "FrequencyMatrix.pkl")
		"""
		tmp_dict = load_pkl(path)
		self.__dict__.clear()
		self.__dict__.update(tmp_dict) 


class SimilarityMatrix:
	"""
	build similarity matrix

		- example:
		
			>>> sim = SimilarityMatrix(X)
			>>> sim.build("jsd")
			>>> W = sim.W
	"""
	def __init__(self, X):
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
		path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+path
		save_pkl(path, self.__dict__)

	def load(self, path="SimilarityMatrix.pkl"):
		"""
		load class from disk

		@param path: file to load (default: "SimilarityMatrix.pkl")
		"""
		tmp_dict = load_pkl(path)
		self.__dict__.clear()
		self.__dict__.update(tmp_dict)

"""
TODO:

- semanticmodel: metadata management
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
			>>> sem.lda(num_topics=100)
			>>> sem.save("romankorpus_lda")
	"""
	def __init__(self, corpus_path):
		self.corpus_path = corpus_path
		self.corpus_name = corpus_path.split("/")[-1]
		self.linesent_path = self.corpus_name+"-LineSentence.txt"
		self.doc_split = None
		self.doc_size = None
		self.freq_filter = None
		self.freq_threshold = None  # percentage of all vocabulary items
		self.freq_min = None
		self.freq_extremes = []
		self.stopword_filter = None
		self.stopword_path = None
		self.stopwords = []
		self.pos_filter = None
		self.pos_tags = None
		self.pos_column = None
		self.bow_dictionary = None
		self.bow_corpus = None
		self.model = None
		self.doc_labels = []  # metadata
		self.info = []  # take note of model parameters

	def preproc(self, doc_split=False, doc_size=1000, freq_filter=False, freq_threshold=0.0001, freq_min=1, stopword_filter=False, stopword_path="stopwords.txt", pos_filter=False, pos_tags=["ADJ", "NN", "V"], pos_column="CPOS"):
		"""
		preprocessing

		@param doc_size: slice input into documents of doc_size words (default: 1000)
		@param freq_threshold: percentage of vocabulary to prune, top and bottom (default: 0.0001)
		@param freq_min: minimum number of occurences for a word to be included (default: 1)
		@param stopword_path: path to stopwords in plaintext file
		@param pos_tags: list of str, use only these POS tags
		"""
		self.doc_split = doc_split
		self.doc_size = doc_size
		self.freq_filter = freq_filter
		self.freq_threshold = freq_threshold
		self.freq_min = freq_min
		self.stopword_filter = stopword_filter
		self.stopword_path = stopword_path
		self.pos_filter = pos_filter
		self.pos_tags = pos_tags
		self.pos_column = pos_column

		# TODO: if the corpus is very large, consume the generator directly into a bow representation
		docs = list(self._iter_docs())

		# filter extremes and stopwords
		c = Counter([word for word in doc for doc in docs])
		self.freq_extremes += set(k for k, v in c.items() if v == self.freq_min)  # minimum number of occurences

		if self.freq_filter is True:
			number_omit_items = int(len(c) * self.freq_threshold)
			self.freq_extremes += c.most_common(number_omit_items)
			self.freq_extremes += c.most_common()[:-number_omit_items-1:-1]  # least common

		if stopword_filter is True:
			with open(self.stopword_path, 'r') as f: sw = f.read()
			self.stopwords += sorted(set(sw.split("\n")))

		if self.freq_filter is True and self.stopword_filter is True:
			docs = [[word for word in doc if word not in self.stopwords if word not in self.freq_extremes] for doc in docs]

		elif self.freq_filter is True and self.stopword_filter is False:
			docs = [[word for word in doc if word not in self.freq_extremes] for doc in docs]

		elif self.freq_filter is False and self.stopword_filter is True:
			docs = [[word for word in doc if word not in self.stopwords] for doc in docs]

		# vectorize
		dictionary = Dictionary(docs)
		corpus = [dictionary.doc2bow(doc) for doc in docs]

		self.bow_dictionary = dictionary
		self.bow_corpus = corpus  # for lda, lsi		
		
		# for word2vec, fasttext: save corpus in LineSentence format
		# cf. https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence
		with open(self.linesent_path, 'w') as f:
			for doc in docs:
				for line in " ".join(doc).split("."):
					f.write(line+"\n")

	def _iter_docs(self):
		paths = [p for p in glob(self.corpus_path) if not p.startswith(".")]
		for p in paths:

			if p.endswith(".txt"):
				with open(p, "r") as f:
					if self.doc_split is True:
						docs_tmp = []
						doc = strip_symbols(f.read()).split(" ")

						while len(doc) >= self.doc_size:
							docs_tmp.append(doc[:self.doc_size])
							del doc[:self.doc_size]
						docs_tmp.append(doc)  # add rest
						
						for i, d in enumerate(docs_tmp, start=1):
							self.doc_labels.append(basename(p).split(".")[0]+" #"+str(i))
							yield d
					else:
						doc = strip_symbols(f.read()).split(" ")
						self.doc_labels.append(basename(p).split(".")[0])  # metadata, e.g. used as plot labels
						yield doc

			elif p.endswith(".csv") or p.endswith(".tsv"):
				df = pd.read_csv(p, sep="\t")  #, quoting=csv.QUOTE_NONE)
	
				if self.pos_filter is True:
					df = df.groupby(self.pos_column)
					doc_df = pd.DataFrame()
					for t in self.pos_tags:
						doc_df = doc.append(df.get_group(t))

					# cheap way to filter out proper names
					#names = df.get_group('NP')["Token"].values.astype(str)
					#self.stopwords += names.tolist()
				else:
					doc_df = df

				# construct documents
				if "Lemma" in doc_df.columns:
					token_column = "Lemma"
				else:
					token_column = "Token"

				if self.doc_split is True:
					doc_df = doc_df.sort(columns='TokenId')
					i = 1
					while(self.doc_size < doc_df.shape[0]):
						doc_str = " ".join(doc_df[:self.doc_size][token_column].values.astype(str))
						doc = strip_symbols(doc_str).split(" ")
						self.doc_labels.append(basename(p).split(".")[0]+" #"+str(i))  # metadata, e.g. used as plot labels
						doc_df = doc_df.drop(doc_df.index[:self.doc_size])  # drop doc_size rows
						i += 1
						yield doc

					doc_str = " ".join(doc_df[token_column].values.astype(str))  # add rest
					self.doc_labels.append(basename(p).split(".")[0]+" #"+str(i))
				else:
					doc_str = " ".join(doc_df[token_column].values.astype(str))
					self.doc_labels.append(basename(p).split(".")[0]+" #"+str(i))
				
				doc = strip_symbols(doc_str).split(" ")
				yield doc

	def lda(self, num_topics=100, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False):
		"""
		lda

		cf. https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
		"""
		self.model = LdaModel(corpus=self.bow_corpus, id2word=self.bow_dictionary, num_topics=num_topics, chunksize=chunksize, passes=passes, update_every=update_every, alpha=alpha, eta=eta, decay=decay, offset=offset, eval_every=eval_every, iterations=iterations, gamma_threshold=gamma_threshold, minimum_probability=minimum_probability, random_state=random_state, ns_conf=ns_conf, minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics)

		# save model parameters
		frame = currentframe()
		args, _, _, values = getargvalues(frame)
		self.info = [a+"="+v for a, v in zip(args, values)]

	def lsa(self, num_topics=200, chunksize=20000, decay=1.0, onepass=True, power_iters=2, extra_samples=100):
		"""
		lsa

		cf. https://radimrehurek.com/gensim/models/lsimodel.html#gensim.models.lsimodel.LsiModel
		"""
		self.model = LsiModel(corpus=self.bow_corpus, id2word=self.bow_dictionary, num_topics=num_topics, chunksize=chunksize, decay=decay, onepass=onepass, power_iters=power_iters, extra_samples=extra_samples)

		# save model parameters
		frame = currentframe()
		args, _, _, values = getargvalues(frame)
		self.info = [a+"="+v for a, v in zip(args, values)]

	def word2vec(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, max_final_vocab=None):
		"""
		word2vec

		cf. https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
		"""
		self.model = Word2Vec(corpus_file=self.linesent_path, size=size, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent, cbow_mean=cbow_mean, iter=iter, null_word=null_word, trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words, compute_loss=compute_loss, max_final_vocab=max_final_vocab)

		# save model parameters
		frame = currentframe()
		args, _, _, values = getargvalues(frame)
		self.info = [a+"="+v for a, v in zip(args, values)]

	def fasttext(self, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, word_ngrams=1, sample=0.001, seed=1, workers=3, min_alpha=0.0001, negative=5, ns_exponent=0.75, cbow_mean=1, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=10000):
		"""
		fasttext

		cf. https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
		"""
		self.model = FastText(corpus_file=self.linesent_path, sg=sg, hs=hs, size=size, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, word_ngrams=word_ngrams, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha, negative=negative, ns_exponent=ns_exponent, cbow_mean=cbow_mean, iter=iter, null_word=null_word, min_n=min_n, max_n=max_n, sorted_vocab=sorted_vocab, bucket=bucket, trim_rule=trim_rule, batch_words=batch_words)

		# save model parameters
		frame = currentframe()
		args, _, _, values = getargvalues(frame)
		self.info = [a+"="+v for a, v in zip(args, values)]

	def save(self, path="SemanticModel.pkl"):
		"""
		save class to disk

		@param path: file to save (default: "SemanticModel.pkl")
		"""
		path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+path
		save_pkl(path, self.__dict__)

	def load(self, path="SemanticModel.pkl"):
		"""
		load class from disk

		@param path: file to load (default: "SemanticModel.pkl")
		"""
		tmp_dict = load_pkl(path)
		self.__dict__.clear()
		self.__dict__.update(tmp_dict)


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
