
# tools

python tools package

## todo

- konsolidieren: in funktionen verkapseln + benennen + docstrings schreiben

- quellen
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
    - Diss/code pre: hgfc, met-clf, met-cluster, met-sampler
        - 
    - Tools: platzhalter-funktionen, links in docstrings
        - 


## package struktur

- cf. https://docs.python.org/dev/tutorial/modules.html#packages

- kategorien
    - korpus/input
    - prozessierung
        - csv/matrix manipulationen
        - queries? oder zu korpus?
            - falls input+proc schwer zu trennen -> corpus class
    - visualisierung
    - allg./misc helper

```
tools/
   __init__.py
   proc.py
   corpus.py
   vis.py
   misc.py
```