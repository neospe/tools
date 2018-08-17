
# tools

python tools package

## todo

- konsolidieren: in funktionen verkapseln + benennen + docstrings schreiben
- numpy ndarray basierte corpus/file-objekt klasse inkl. methoden aus den modulen (io, proc, vis)

- quellen -> **TODO** funktionen unten bei modulen einordnen
    - Code/code pre: helper, topic modeling, van halteren, word embedding
        - ctree-reader.py: dof -> nltk tree -> queries
        - dtree-reader.py: dof -> nx graph -> queries
        - reader.py: dof -> queries
        - count_tok.py: dof dir -> count
        - doc_split.py: txt -> txt dir
        - gutenberg_xml_year.py: xml dir -> lxml -> queries
        - stripxml_lexika.py: xml -> txt
        - wordsplitter_dict.py: dof dir -> dict
        - lda.py: dof dir -> filter, norm, vec -> model
        - lda_heatmap.py: model -> sort -> img
        - lda_network.py: model -> nx graph -> img
        - clf_alltokens.py: dof dir -> vec, clf, crossval -> predict
        - vh_tutorial.py: dof dir -> featsel, vec, clf, crossval -> predict
        - doc2vec.py: txt dir -> doc2vec -> tsne, cluster
        - lsa.py: txt dir -> lsa -> tsne, cluster
    - Diss/code pre: met-cluster, met-sampler
        - met-cluster
            - cluster.py, scale_features: np array -> np array
            - cluster.py, plot_graph, plot_bi_graph: nx graph, labels -> img
            - cluster.py, plot_matrix_sorted: np array, labels -> img
            - cluster.py, nx_graph_from_biadjacency_pandas_df: df -> nx graph
            - cluster.py, dist_to_sim: df -> df
            - corpus.py, select_romankorpus: pkl dir, metadaten -> paths list, years list
            - corpus.py, select_gutenberg: pkl dir, xml dir -> paths list, years list
            - corpus.py, count_tokens: dof dir -> int
            - corpus.py, plot_docs_years: years list -> img
            - extract.py, most_frequent_nouns: dof dir -> list
            - extract.py, clean_string: str -> str
            - features.py, filter_min_freq, filter_top_feat, filter_by_mfn, filter_by_pos: lists -> lists (nouns, feats)
            - features.py, jsd: np array, np array -> real
            - features.py, count_features: lists (nouns, feats) -> np array, dicts (nouns, feats)
            - features.py, scale_features: np array -> np array
            - features.py, similarity_matrix: np array -> np array
            - features.py, plot_matrix, save_matrix, load_matrix: csv -> np array -> img
        - met-sampler
            - classify.py, tfidf_calculate: dof dir -> df, clf
            - classify.py, tfidf_classify: clf, df, str -> bool
            - webanno spezifisch
                - sample.py: write_tsv, stratified random sampler for sentences
                - collect.py: substring_match, sent_lookup (levenshtein), sent_search
                - eval.py: plot_candidate_types, count_tokens, plot_met_types, plot_met_annotations, calculate_agreement_annotations
    - Code/fundus -> platzhalter-funktionen, docstrings inkl. beschreibung + links 
        - pickle alternative -> in funktion wrappen und z.b. export() nennen
        - smart_open()

- module
    - io
    
    - proc
    
    - vis
    
    - misc

## package struktur

- cf. https://docs.python.org/dev/tutorial/modules.html#packages

```
tools/
   __init__.py
   io.py
   proc.py
   vis.py
   misc.py
```