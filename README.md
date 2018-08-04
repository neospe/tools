
# tools

Digital Philologist's toolbox.

## what's inside

- **proc**: Processing tools and classifiers.
  - **clf**: General classification utilities.
  - **authorship clf**: Classifier for determining likely authorship of texts.
  - **freq matrix**: Computes frequency matrices from text data.
  - **metaphor sentence clf**: Identifies metaphorical expressions in sentences.
  - **semantic models**: Implements semantic vector space models (e.g. word embeddings).
  - **sentence clf**: Classifies sentences based on predefined criteria.
  - **sim matrix**: Calculates sentence or word similarity matrices.
  - **tfidf sentence clf**: Applies TF-IDF methods for sentence-level classification.
  - **token clf**: Classifies individual tokens (e.g., for part-of-speech tagging).
- **vis**: Visualization tools for data, models, or classification results.
- **io**: Handles input/output operations.
- **misc**: Miscellaneous utilities.

See the [documentation](apidocs/index.html) for more.

## building the documentation

- _pip install pydoctor_
- _pydoctor --add-package .__

### optional: convert to dash

- _pip install doc2dash_
- _doc2dash apidocs_

### docstring format

- epytext: http://epydoc.sourceforge.net/manual-epytext.html
- http://epydoc.sourceforge.net/fields.html

