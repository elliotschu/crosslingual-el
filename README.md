# Cross-Lingual Entity Linker
This repository contains an implementation of a cross-lingual entity linker, using Multilingual BERT (or, any transformer you like!).

The architecture is described in _Cross-Lingual Transfer in Zero-Shot Cross-Language Entity Linking_, Schumacher et al 2020 (to Appear in Findings of ACL 2021, https://arxiv.org/abs/2010.09828).

# Use
To run this project locally, first clone it:

```bash
git clone https://gitlab.hltcoe.jhu.edu/eschumacher/clel.git
```

and add the repository to your python path.

A list of requirements is available in `requirements.txt`.

Then enter and modify the `config.ini` file. Relevant configurations you may want to change are as follows:

- `root`: a directory to cache results during both training and prediction (and save models).

- `data_directory`: the directory where the input files (and, possibly, the ontology) are located

-`dataset`: Either `Wiki` or `LDC2019T02` are currently supported.

# Data

The linker takes the following objects as input;

- `mentions`, a `data/Objects.py/Mentions` object, containing all training mentions
- `evaluation_mentions`, a `data/Objects.py/Mentions` object, containing all evaluation mentions
- `ontology`, a `data/Objects.py/Mentions` object, containing all entities.  This includes the training and evaluation gold standard entities and any candidate entities.

Each mention object needs a SpaCy document with the mention span extracted, in addition to metadata information (e.g. the gold standard entity).
In addition, a list of candidate entities needs to be included - the linker will only score those candidates at prediction.

We include, in the data directory, readers for two datasets -- LDC2019T02 (which can be downloaded here https://catalog.ldc.upenn.edu/LDC2019T02), 
and a subset of Wikipedia, included in this repository.

For candidate generation, we use https://github.com/shyamupa/wiki_candgen.  After generating candidate files, place them in the data directory.

###Wiki
We sampled a portion of the dataset provided by Pan et al 2017 (https://www.aclweb.org/anthology/P17-1178/).

All of the data required is available in this repository, and is structured as follows;
```
Wiki
+-- ar
    +-- mention.ar.cands.csv (candidates included)
    +-- mention.ar.csv
    +-- wiki.ar.cands.pkl (includes cached wikipedia candidate pages)
    +-- wiki_ar.pkl
(Structure repeated for all languages)
+-- ru
+-- fa
+-- ko

```    

###LDC2019T02
If using this dataset, acquire from the LDC (we cannot redistribute it ourselves), and point the `data_directory` to a directory structured as follows
```
LDC2019T02
+-- CMN
    +-- eval (containing all eval files)
    +-- training (containing all training files)
    +-- tac_kbp_2015_tedl_eval_gold_standard_entity_mentions.tab
    +-- tac_kbp_2015_tedl_training_gold_standard_entity_mentions.tab
    +-- mentions_CMN_out.csv (candidate entity file)
    +-- mentions_CMN__out_eval.csv (candidate entity file)
(Structure repeated for all languages)
+-- SPA
+-- ENG
```    

For this dataset (and this dataset only), the FreebaseTools 1.20 toolkit is used from the ISI to process the Freebase KB.   This is only used within `data/LDC2019T02.py`. The link we downloaded this from is currently unavailable,
and we'd suggest that if intending to use LDC2019T02, use a different toolkit for accessing the KB.


