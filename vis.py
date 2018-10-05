"""
visualisation
"""

def plot_matrix(X, outfile):
	"""
	plot matrix

	cf. met-cluster/features.py
	"""
	plt.figure(figsize=(80,80))
	plt.matshow(X, fignum=100, cmap=cm.gray)
	plt.savefig(outfile, dpi=160)
	"""
	alternativ:

	plt.title("Original dataset")
	plt.matshow(X, cmap=plt.cm.Blues)
	plt.xlabel(sys.argv[1])
	plt.savefig(outdir+'/data.png', dpi=90)
	"""


def plot_matrix_sorted(W, B, clust_labels, outfile):
	"""
	plot matrix sorted by clustering + silhouette-coefficient

	cf. met-cluster/cluster.py
	"""
	# http://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
	# The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
	# Scores around zero indicate overlapping clusters. The score is higher when clusters are dense
	# and well separated, which relates to a standard concept of a cluster.
	score = metrics.silhouette_score(W, np.asarray(clust_labels), metric='precomputed')     # 2015 ergebnisse waren mit 'cosine'

	# sort original data by cluster labels
	fit_data = W[np.argsort(clust_labels)]
	fit_data = fit_data[:, np.argsort(clust_labels)]

	# plot matrix
	plt.figure(figsize=(80,80))
	plt.matshow(fit_data, fignum=100, cmap=plt.cm.Blues)
	plt.xlabel('clusters: '+str(B.shape[1])+'\nsilhouette score: '+str(score), fontsize=12)
	plt.savefig(outfile, dpi=160)

	"""
	alternativ:

	# spectral bi+coclustering: arrange to show checkerboard structure
	plt.matshow(np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1), cmap=plt.cm.Blues)
	plt.title("Checkerboard structure of rearranged data")
	"""


def plot_graph(G, labels, fileout):
	"""
	plot networkx graph

	cf. met-cluster/cluster.py
	"""
	plt.figure(figsize=(80,80))
	pos = nx.spring_layout(G, k=0.2, iterations=100)   #, scale=10.0)
	#pos = nx.shell_layout(G)
	#pos = nx.circular_layout(G)
	nx.draw(G, pos, node_size=20, alpha=0.3, node_color='blue', with_labels=True)
	#nx.draw_networkx_labels(G, pos, labels=labels, font_color='g')
	plt.savefig(fileout, dpi=160)
	"""
	alternativ:

	G = nx.from_pandas_dataframe(sim, '0', '1', edge_attr='2')
	#G_bi = nx.make_clique_bipartite(G)

	plt.figure(figsize=(30,30))
	pos = nx.spring_layout(G_bi)
	nx.draw(G, pos, node_size=20, alpha=0.1, node_color='blue', with_labels=False)
	nx.draw_networkx_labels(G,pos,font_color='g')
	plt.savefig('graph.png')
	"""


def plot_bi_graph(B, labels, fileout):
	"""
	plot networkx bipartite graph

	cf. met-cluster/cluster.py
	"""
	X, Y = nx.bipartite.sets(B)
	plt.figure(figsize=(12,12))
	pos = dict()
	pos.update( (n, (1, i+20)) for i, n in enumerate(X) ) # put nodes from X at x=1
	pos.update( (n, (2, i+20)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
	nx.draw_networkx(B, pos=pos, node_size=70, font_size=10, alpha=0.5, width=0.1)
	#nx.draw_networkx_labels(B, pos, labels=labels, font_color='g')
	plt.savefig(fileout, dpi=160)


def plot_histogram(years):
	"""
	plot histogram

	cf. met-cluster/corpus.py
	"""
	data = pd.DataFrame.from_dict(data=Counter(years), orient='index')
	#data = data.sort_index(ascending=False)
	#data.sort_values(0, axis='index', ascending=False, inplace=True)
	data.sort_index(inplace=True)

	# bin dataframe
	data['year'] = data.index.astype(int)
	bins = np.arange(1800, 1950, 10)
	labels = [str(i) for i in np.arange(1800, 1940, 10)]

	data['bin'] = pd.cut(data['year'], bins=bins, labels=labels, include_lowest=True)
	data['bin'] = data['bin'].astype(int)

	grouped = data.groupby('bin')
	#data = grouped.agg(np.sum)     # TypeError: unorderable types: str() < int()

	# alternativ zu agg()
	size_per_bin = {}
	for g, sizes in grouped: size_per_bin[g] = sizes[0].sum()
	data = pd.DataFrame.from_dict(data=size_per_bin, orient='index')

	data.sort_index(inplace=True)

	# plot
	plt.figure()
	ax = sns.barplot(data=data.T, orient='h', palette='PuBuGn_d')  # data=data[:25].T
	ax.set_title('Novels by creation date')
	ax.set(xlabel=str(data[0].sum())+' works total\n'+format_number(tok_count)+' tokens', ylabel='')

	plt.savefig('works-genre.png', dpi=80)

	"""
	fuer diskrete daten:

	data = pd.DataFrame.from_dict(data=Counter(count), orient='index')
	#data = data.sort_index()
	data.sort_values(0, axis='index', ascending=False, inplace=True)
	#data.columns = ['']

	plt.figure()
	plt.tick_params(axis='both', which='major', labelsize=15)
	#data.plot.pie(figsize=(6, 6), subplots=True, colormap='Blues')
	ax = sns.barplot(data=data.T, orient='h', palette='PuBuGn_d')  # data=data[:25].T
	ax.set_title('top features')
	plt.savefig(result_path+'/'+result_file[:-4]+'_topfeat.png', dpi=80)
	"""


def plot_dendrogram(X, outfile):
	"""
	plot dendrogram
	
	cf. met-cluster/cluster.py
	"""
	Z = scipy.cluster.hierarchy.ward(X)
	tree = scipy.cluster.hierarchy.to_tree(Z)   # for tree-based traversal of linkage matrix
    
	level = 200
	plt.figure(figsize=(240,240))

	dendro = dendrogram(Z, level, orientation='left', labels=labels, show_leaf_counts=True, leaf_label_func=llf, leaf_rotation=90, leaf_font_size=1, truncate_mode='level', distance_sort='ascending', count_sort='ascending')

	plt.xlim([0, 500])                          # truncate to stay within size limit even for high levels
	plt.savefig('out/dendrogram_lvl'+str(level)+'.png', dpi=90)


################
# DENDROGRAM LABELING

# cf. hgfc/cluster.py
# cf. http://rootslash.net/20808/scipy-hierarchical-cluster-dendrogram-labels-mixed-up

# when given a tree and id, it returns all the leave nodes
# (indices for nouns) to retrieve the nouns of that cluster
def search_tree(tree, id):
    if tree.get_id() == id:
        return tree.pre_order()
    if tree.is_leaf():
        return []
    return search_tree(tree.get_left(), id) + search_tree(tree.get_right(), id)

# leaf label function
# maps the indices back to the original nouns
def llf(id):
    if id < labels_len:
        label = str(labels[id]) + '\n'
        return label
    else:
        indices = search_tree(tree, id)
        nouns = [labels[index] for index in indices][0:30]      # display only the first 30 nouns
        output = [str(len(nouns)) + ":"] + nouns
        return ' '.join(output)

# same as llf() but returns a list instead of a label string
def label_lookup(id):
    if id < labels_len:
        label = [str(labels[id])]
        return label
    else:
        indices = search_tree(tree, id)
        nouns = [labels[index] for index in indices]
        return nouns

################


def tsne_cluster(X, outfile):
	"""
	word embedding/doc2vec.py + lsa.py: model -> tsne, cluster
	"""

	"""
	alternativ: 
	cf. met-cluster/cluster.py

	# scipy agg / sklearn agg 3d

	#projection = TSNE(n_components=3)
	#projection.fit(X)

	agg_labels = scipy.cluster.hierarchy.fcluster(Z, 800, criterion="maxclust")
	#agg_labels = AgglomerativeClustering(100).fit_predict(X)

	logging.debug('shape of labels array: ', agg_labels.shape)

	fig = plt.figure(figsize=(40, 40), dpi=80)
	ax = fig.add_subplot(111, projection='3d')
	palette = sns.palettes.color_palette('spectral', 850)

	for j in set(agg_labels):
	    ax.scatter(X_3d[agg_labels == j, 0], X_3d[agg_labels == j, 1], X_3d[agg_labels == j, 2],
	                s=50, color=palette[j], alpha=0.5, label=labels[j], zdir=u'y')
	#plt.title('t-SNE embedding with AgglomerativeClustering labels')
	#plt.legend()
	#plt.savefig('out/scatter3d_aggscipy.png', dpi=80)
	plt.show()
	"""
	

def lda_heatmap(X, outfile):
	"""
	topic modeling/lda_heatmap.py: model -> sort -> img
	"""
	

def lda_network(X, outfile):
	"""
	topic modeling/lda_network.py: model -> nx graph -> img
	"""
	

def scatterplot_decision_surface(X, outfile):
	"""
	van halteren/vh_tutorial.py: clf decision surfaces -> img 
	"""


def plot_fuzzyclustering(X, outfile):
	"""
	cf. met-cluster/cluster.py
	"""

	"""
	import skfuzzy as fuzz

	ncenters = 80
	palette = sns.palettes.color_palette('spectral', 85)

	cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, ncenters, 2, error=0.005, maxiter=1000, init=None)

	# Plot assigned clusters, for each data point in training set
	cluster_membership = np.argmax(u, axis=0)

	#logging.debug(type(cluster_membership))
	logging.debug(cluster_membership.shape)

	for j in range(ncenters):
	    plt.plot(X_2d[cluster_membership == j, 0],
	             X_2d[cluster_membership == j, 1], '.', color=palette[j])

	# Mark the center of each fuzzy cluster
	for pt in cntr:
	    plt.plot(pt[0], pt[1], 'rs')

	#plt.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
	#plt.axis('off')

	plt.tight_layout()
	plt.savefig('out/scatter_fuzzy.png', dpi=80)
	"""


def scatterplot_hexbin(X, outfile):
	"""
	cf. met-cluster/cluster.py
	"""

	"""
	# hexbin + scatterplot (tsne projection of original data)

	logging.debug('plotting hexbin ..\n')

	plt.figure()
	plt.title("Density")
	plt.hexbin(*X_2d.T)

	plt.savefig('out/hexbin.png', dpi=160)


	logging.debug('plotting scatterplot ..\n')

	assignments = scipy.cluster.hierarchy.fcluster(Z, 4, criterion="maxclust")
	#logging.debug(scipy.cluster.hierarchy.leaders(Z, assignments))

	plt.figure()
	plt.title("4 state ward clustering of data")
	plt.plot(X_2d[assignments==1, 0], X_2d[assignments==1, 1], 'o', label="1")
	plt.plot(X_2d[assignments==2, 0], X_2d[assignments==2, 1], 'o', label="2")
	plt.plot(X_2d[assignments==3, 0], X_2d[assignments==3, 1], 'o', label="3")
	plt.plot(X_2d[assignments==4, 0], X_2d[assignments==4, 1], 'o', label="4")
	plt.legend()

	plt.savefig('out/scatter.png', dpi=160)
	"""


def scatterplot_hdbscan(X, outfile):
	"""
	cf. met-cluster/cluster.py
	"""

	"""
	# hdbscan

	hdbscan_labels = hdbscan.HDBSCAN().fit_predict(X)
	palette = sns.palettes.color_palette('spectral', 100)
	for digit in set(map(int,hdbscan_labels)):
	    if digit == -1:
	        plt.scatter(projection.embedding_.T[0][hdbscan_labels == digit],
	                    projection.embedding_.T[1][hdbscan_labels == digit],
	                    color='black', alpha=0.5, label='unclassified')
	    else:
	        plt.scatter(projection.embedding_.T[0][hdbscan_labels == digit],
	                    projection.embedding_.T[1][hdbscan_labels == digit],
	                    color=palette[digit], alpha=0.5, label=str(digit))
	plt.title('t-SNE embedding with HDBSCAN labels')
	plt.legend()
	plt.savefig('out/scatter_hdb.png', dpi=160)
	"""


def scatterplot_3d(X, outfile):
	"""
	cf. met-cluster/cluster.py
	"""

	"""
	# scatterplot 3d (tsne projection of original data)

	fig = plt.figure(num=1, figsize=(32, 24), dpi=80, facecolor="w", edgecolor="k")
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], zdir=u'y', s=50)

	#plt.savefig('out/scatter3d.png', dpi=160)
	plt.show()
	"""


def matrix_dendrogram(X, outfile):
	"""
	cf. met-cluster/cluster.py
	"""
	
	"""
	# scipy agg matrix plot

	import scipy
	import pylab
	import scipy.cluster.hierarchy as sch

	level = 100

	# Generate features and distance matrix.
	#x = scipy.rand(40)
	#D = scipy.zeros([40,40])
	#for i in range(40):
	#    for j in range(40):
	#        D[i,j] = abs(x[i] - x[j])

	# Compute and plot dendrogram.
	fig = pylab.figure(figsize=(80,80))
	axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
	#Y = sch.linkage(D, method='centroid')
	#dendro = sch.dendrogram(Z, level, orientation='right', truncate_mode='level', distance_sort='ascending', count_sort='ascending')
	dendro = sch.dendrogram(Z, orientation='right', distance_sort='ascending', count_sort='ascending')
	axdendro.set_xticks([])
	axdendro.set_yticks([])

	# Plot distance matrix.
	axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])

	index = dendro['leaves']
	#index = list(set(dendro['leaves']))

	#logging.debug(type(index))
	#logging.debug(index)

	X = X[index,:]
	X = X[:,index]

	im = axmatrix.matshow(X, aspect='auto', origin='lower')
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])

	# Plot colorbar.
	axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
	pylab.colorbar(im, cax=axcolor)

	# Display and save figure.
	fig.show()
	fig.savefig('out/matrix_dendro.png', dpi=160)
	"""