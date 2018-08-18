
# about

eine sammlung von funktionen, die der erinnerung an oft-, wenig- und noch-nicht-genützte lösungswege dienen soll.

## todo

- docstrings schreiben
- neue komponenten
    - export funktion: pickle alternative + state saving mechanismus (z.b. parameter/variablen -> json)
    - platzhalter-funktionen aus code fundus: docstrings inkl. beschreibung + links

## docset workflow

- in bash + python 2.7: _source activate py2_
- pydoctor (https://launchpad.net/pydoctor):
    - install: _pip install pydoctor_
        - edit template: ~/miniconda3/envs/py2/lib/python2.7/site-packages/pydoctor/templates/common.html
            - remove div id="showPrivate"
    - _pydoctor --add-package path-to-package_
- doc2dash (https://pypi.org/project/doc2dash):
    - install: _conda install --channel "conda-forge" doc2dash_
    - _doc2dash path-to-apidocs_

## module

- io
    - met-cluster/features.py, save_matrix, load_matrix: csv <-> np array
    - met-cluster/extract.py, clean_string: str -> str
    - helper/stripxml_lexika.py: xml -> txt
    - helper/reader.py: dof -> queries
    - helper/gutenberg_xml_year.py: xml dir -> lxml -> queries
    - helper/ctree-reader.py: dof -> nltk tree -> queries
    - helper/dtree-reader.py: dof -> nx graph -> queries
    - met-cluster/cluster.py, nx_graph_from_biadjacency_pandas_df: df -> nx graph
- proc
    - helper/count_tok.py: dof dir -> int
    - met-cluster/extract.py, most_frequent_nouns: dof dir -> list
    - helper/wordsplitter_dict.py: dof dir -> dict
    - met-cluster/features.py, count_features: lists (nouns, feats) -> np array, dicts (nouns, feats)
    - met-cluster/features.py, scale_features: np array -> np array
    - met-cluster/cluster.py, scale_features: np array -> np array
    - met-cluster/features.py, filter_min_freq, filter_top_feat, filter_by_mfn, filter_by_pos: lists -> lists (nouns, feats)
    - met-cluster/features.py, jsd: np array, np array -> real
    - met-cluster/features.py, similarity_matrix: np array -> np array
    - met-cluster/cluster.py, dist_to_sim: df -> df
    - topic modeling/lda.py: dof dir -> filter, norm, vec -> model
    - van halteren/clf_alltokens.py: dof dir -> vec, clf, crossval -> predict
    - van halteren/vh_tutorial.py: dof dir -> featsel, vec, clf, crossval -> predict
    - met-sampler/classify.py, tfidf_calculate: dof dir -> df, clf
    - met-sampler/classify.py, tfidf_classify: clf, df, str -> bool
    - tlw. zu vis
        - word embedding/doc2vec.py: txt dir -> doc2vec -> tsne, cluster
        - word embedding/lsa.py: txt dir -> lsa -> tsne, cluster
- vis
    - met-cluster/features.py, plot_matrix: np array -> img
    - topic modeling/lda_heatmap.py: model -> sort -> img
    - topic modeling/lda_network.py: model -> nx graph -> img
    - met-cluster/cluster.py, plot_graph, plot_bi_graph: nx graph, labels -> img
    - met-cluster/cluster.py, plot_matrix_sorted: np array, labels -> img
    - met-cluster/corpus.py, plot_docs_years: years list -> img
- misc
    - helper/doc_split.py: txt -> txt dir
    - met-cluster/corpus.py, select_romankorpus: pkl dir, metadaten -> paths list, years list
    - met-cluster/corpus.py, select_gutenberg: pkl dir, xml dir -> paths list, years list
    - webanno spezifisch
        - met-sampler/sample.py: write_tsv, stratified random sampler for sentences
        - met-sampler/collect.py: substring_match, sent_lookup (levenshtein), sent_search
        - met-sampler/eval.py: plot_candidate_types, count_tokens, plot_met_types, plot_met_annotations, calculate_agreement_annotations

## quellen

- Code/code pre: helper, topic modeling, van halteren, word embedding
- Diss/code pre: met-cluster, met-sampler
- Code/fundus

## package struktur

- cf. https://docs.python.org/dev/tutorial/modules.html#packages

```
tools/
   __init__.py
   io.py
   misc.py
   proc.py
   vis.py
```