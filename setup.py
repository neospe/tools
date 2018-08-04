from setuptools import setup

setup(
   name='tools',
   version='0.1',
   description='Digital Philologist's toolbox',
   author='Stefan Pernes',
   author_email='stefan@pernes.net',
   packages=['tools'],
   install_requires=[
      'scikit-learn',
      'json-tricks',
      'nltk',
      'lxml',
      'networkx',
      'numpy',
      'pandas',
      'joblib',
      'scipy',
      'gensim',
      'seaborn',
      'matplotlib'
   ]
)
