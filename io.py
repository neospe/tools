"""
input/output
"""

import pandas as pd
from re import sub
from sklearn.externals import joblib
from json_tricks.np import dump, load
from json_tricks.encoders import pandas_encode, numpy_encode
from nltk.tree import ParentedTree
from nltk.corpus.reader.xmldocs import XMLCorpusView
from lxml import etree
from networkx import Graph, DiGraph


def save_pkl(path_or_buf, objects):
	"""
	save objects
	
		- example:
		
			>>> save_pkl("file.pkl", [obj1, obj2, obj3])

	@param path_or_buf: path or file object
	@param objects: list of objects
	"""
	joblib.dump(objects, path_or_buf, compress=1)


def load_pkl(path_or_buf):
	"""
	load objects
	
		- example:
		
			>>> [obj1, obj2, obj3] = load_pkl("file.pkl")

	@param path_or_buf: path or file object
	@return: list of objects
	"""
	return joblib.load(path_or_buf)


def save_json(path_or_buf, objects):
	"""
	save objects to json file

	supports primitives, pandas and numpy objects, custom classes (cf. https://json-tricks.readthedocs.io/en/latest)

	no nested complex types
	
		- example:
		
			>>> save_json("file.json", [obj1, obj2, obj3])

	@param path_or_buf: path or file object
	@param objects: list of objects
	"""
	dump(objects, path_or_buf, obj_encoders=[pandas_encode(content), numpy_encode(content)], allow_nan=True)


def load_json(path_or_buf):
	"""
	load objects from json file
	
		- example:
		
			>>> [obj1, obj2, obj3] = load_json("file.json")

	@param path_or_buf: path or file object
	@return: list of objects
	"""
	return load(path_or_buf)


def strip_symbols(string):
	"""
	remove special characters from string

	@param string: string to modify
	@return: string
	"""
	return sub('[\W_]+', '', string)


def strip_xml(path):
	"""
	export text content from xml file
	
	@param path: path to file
	"""
	result = XMLCorpusView(path,".*/text/body/div")
	result_iterator = result.iterate_from(0)
	
	f = open(path+".txt", "w")

	for element in result_iterator:
		f.write(str(element[0].text) + " " + str(element[1].text) + "\n\n")

	f.close()


def query_xml(path, xpath, out_type="element"):
	"""
	query xml document

	returns a list of elements (cf. https://lxml.de/xpathxslt.html) or strings
	
		- example:
		
			>>> query_xml("mytext.xml", "//head/meta[@name='year']")

	@param path: path to file
	@param xpath: xpath
	@param out_type: "element", "text" or "attrib" (default: "element")
	@return: list
	"""
	with open(path, 'r', encoding='latin1') as f:
		doc = etree.parse(f)

	if out_type == "text":
		return [result.text for result in doc.xpath(xpath)]
	elif out_type == "attrib":
		return [result.attrib['content'] for result in doc.xpath(xpath)]
	else:
		return doc.xpath(xpath)


def query_df(df, column, values):
	"""
	query dataframe
	
		- example:
			
			>>> query_df(df, "CPOS", ["ADJ", "NN"])

	@param df: dataframe
	@param column: target column
	@param values: list of one or more values to match
	@return: dataframe
	"""
	return df.loc[df[column].isin(values)]


def ctree_from_df(df):
	"""
	extract constituency tree from sentence df
	
	@param df: dataframe for one sentence (CoNLL 2009/DOF)
	@return: nltk.tree.ParentedTree
	"""
	delim = '#'          # token#token_id delimiter for use inside tree objects
	sent_string = ""

	for row in df.iterrows():
		tok_id = str(row[0])                                            # current token id
		tok = row[1].get("Token")                                       # current token
		tree_frag = row[1].get("SyntaxTree").strip("*")                 # current syntax tree fragment

		"""
		if "*)" in sent_string:                                         # TODO: possible bug in csv tree writer
			if tmp_string:
				tmp_string2 = sent_string.replace("*)", "")             # we ran into "*)" a second time
				sent_string = tmp_string2
			else:
				tmp_string = sent_string.replace("*)", "")
				sent_string = tmp_string
		"""

		if tree_frag.startswith("("):                                   # reconstruct tree + save token id
			sent_string += tree_frag + " " + tok + delim + tok_id + " "  # beginning of fragment
		elif not ")" in tree_frag:
			sent_string += tree_frag + " " + tok + delim + tok_id + " "  # middle
		else:                                                           # end
			"""
			if tmp_string:                                              # TODO: possible bug in csv tree writer
				sent_string += " " + tok + delim + tok_id + tree_frag + ") "
				tmp_string = ""
			elif tmp_string2:
				sent_string += " " + tok + delim + tok_id + tree_frag + ")) "
				tmp_string2 = ""
			else:
			"""
			sent_string += " " + tok + delim + tok_id + tree_frag + " "

	return ParentedTree.fromstring(sent_string)


def query_ctree(tree, values):
	"""
	query constituency tree

	matches labels and leaves, returns list of subtrees

		- example:
			
			>>> query_ctree(tree, ["NP", "Effi"])

	@param tree: ParentedTree for one sentence
	@param values: list of one or more values to match
	@return: list
	"""
	trees = [s for s in tree.subtrees(filter=lambda t: t.label() in values)]
	trees += [s for s in tree.subtrees if any(v in s.leaves() for v in values)]

	return trees


def dtree_from_df(df):
	"""
	extract dependency tree from sentence df
	
	@param df: dataframe for one sentence (CoNLL 2009/DOF)
	@return: nx.DiGraph
	"""
	dg = DiGraph()                                          # a new directed graph

	for row in df.iterrows():
		tok_id = str(row[0])                                # current token id
		tok = row[1].get("Token")                           # current token
		head_id = row[1].get("DependencyHead")              # token head id
		rel = row[1].get("DependencyRelation")              # dependency relation

		if head_id.isdigit() == True:
			head = df.iloc[int(head_id), 6]                 # get head token
		else:
			head = "ROOT"                                   # or mark as root

		dg.add_node(tok, id=tok_id)                         # save token id as node attribute
		dg.add_node(head, id=head_id)
		dg.add_edge(head, tok, rel=rel)                     # add edge to graph

	return dg


def query_dtree(tree, values):
	"""
	query dependency tree

	matches labels and leaves, returns list of nodes

		- example:
			
			>>> query_dtree(tree, ["NK", "von"])

	@param tree: DiGraph for one sentence
	@param values: list of one or more values to match
	@return: list
	"""
	
	# search edges
	relations = dict([((u, v), d['rel']) for u, v, d in tree.edges(data=True)])
	nodes = [node for node, rel in relations.items() if rel in values]

	# search nodes
	tokens = dict([(u, d['id']) for u, d in tree.nodes(data=True)])
	nodes += [node for node, i in tokens.items() if node in values]

	return nodes


def graph_from_biadjacency_df(df):
	"""
	construct bipartite graph from biadjacency matrix
	
	@param df: biadjacency matrix as dataframe
	@return: nx.Graph
	"""
	B = Graph()

	"""
	for i in df.index:
		B.add_node(i, bipartite=0)
		for j in df.columns:
			B.add_node(j, bipartite=1)
			if (df.ix[i,j] > 0):
				B.add_edge(i, j, weight=df.ix[i,j])
	"""
	for i, i_label in zip(range(df.shape[0]), df.index):
		for j, j_label in zip(range(df.shape[1]), df.columns):
			j_label = j_label+'_'
			if df.iloc[i,j] > 0:
				B.add_node(i_label, bipartite=0)
				B.add_node(j_label, bipartite=1)
				B.add_edge(i_label, j_label, weight=df.iloc[i,j])
			#else:
			#    if B.edges(i_label) == None: B.remove_node(i_label)
			#    if B.edges(j_label) == None: B.remove_node(j_label)

	return B


if __name__ == '__main__':
	print("not specified")