"""
input-output module
"""

import pandas as pd

def save_matrix(path, X, nouns_dict, feats_dict):
    print('\nsaving..', end=' ')
    now = time.time()
    dump(nouns_dict, result_path+'/nouns_dict.pkl')

    if feats_dict != None:
        dump(feats_dict, result_path+'/feats_dict.pkl')
        X_df = pd.DataFrame(data=X, index=sorted(feats_dict.keys()), columns=sorted(nouns_dict.keys()), dtype='float16')
    else:
        X_df = pd.DataFrame(data=X, index=sorted(nouns_dict.keys()), columns=sorted(nouns_dict.keys()), dtype='float64')

    X_df.to_csv(path, '\t')
    print('done:', path, X_df.shape, 'in', time.time()-now , 'sec\n')

def load_matrix(path):
    print('\nloading..', end=' ')
    now = time.time()

    nouns_dict = load(result_path+'/nouns_dict.pkl')
    feats_dict = load(result_path+'/feats_dict.pkl')

    X_df = pd.read_csv(path, sep='\t', index_col=0)
    X = X_df.as_matrix()

    print('done:', path, X_df.shape, 'in', time.time()-now , 'sec\n')

    return X, nouns_dict, feats_dict

def clean_string(token):
    strip_chars = '_–-—«»›‹/\§!?.,"„“% '

    if isinstance(token, str):
        new = token.strip(strip_chars).lower()

        if '|' in new:
            new = new.split('|')[0]     # treetagger format

        if len(new) > 1 and '&' not in new:
            if re.match("^[a-z_]*$", new):
                return new
            else:
                return None

def strip_xml(path):
	from nltk.corpus.reader.xmldocs import XMLCorpusView
	
	view = XMLCorpusView(path,'.*/text/body/div')
	iter = view.iterate_from(0)
	f = open(path+'.txt', 'w')

	for entry in iter:
	    f.write(str(entry[0].text) + ' ' + str(entry[1].text) + '\n\n')

	f.close()

def query_df(df, column, match=[value1, value2]):
	"""
	aka reader.py
	
	Args:
	df -- dataframe
	column -- column to select
	match -- list of one or more values to match (optional)

	Returns:
	df -- ataframe

	Examples:
	>>> query_df(df, "CPOS", ["ADJ","NN"])
	"""
	groups = df.groupby(column)
	result = pd.DataFrame()
	for m in match:
		result = pd.concat([result, groups.get_group(m)])

	return result

def query_xml(path, xpath=[path1, path2], match=[]):
	"""
	aka gutenberg_xml_year.py

	path -- path to directory
	xpath -- list of one or more xpaths to execute
	match -- list of one or more values to match (optional)

	returns dataframe
	"""

	return

def query_dof_ctree(df, match):

	return

def query_dof_dtree(df, match):

	return

def nx_graph_from_biadjacency_pandas_df(df):

	return

#if __name__ == '__main__':
#	print("not specified")