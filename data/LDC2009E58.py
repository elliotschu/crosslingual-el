"""
Elliot Schumacher, Johns Hopkins University
Created 10/14/19
"""
import spacy
import pandas as pd
import logging
import os
from spacy.matcher import Matcher
import re
from collections import defaultdict
from io import StringIO, BytesIO
from data.Objects import *
subdirectories = {
    "it": "europarl",
    "da": "europarl",
    "fi": "europarl",
    "nl": "europarl",
    "pt": "europarl",
    "sv": "europarl",

    "es": "proj_synd"
}

def load_ontology(add_nil=True):
    log =logging.getLogger()
    data_directory = "../datasets/LDC2009E58/data"
    ontology = Ontology()
    for filename in os.listdir(data_directory):
        if filename.endswith(".xml"):
            tree = etree.parse(os.path.join(data_directory, filename))
            for entity in tree.getroot().iter("entity"):
                entity_dict = dict(entity.attrib)
                wiki_text = entity.findtext('wiki_text')
                fact_dict = defaultdict(list)
                fact_elements = entity.find('facts').findall('fact')
                for fe in fact_elements:
                    if fe.text:
                        fact_dict[fe.attrib['name']].append(fe.text)
                    link_elements = fe.findall('link')
                    for le in link_elements:
                        if le.text:
                            fact_dict[fe.attrib['name']].append(le.text)
                ontology.add(id=entity_dict['id'],
                             oe=Ontology_Entity(id=entity_dict["id"],
                                              name= entity_dict["name"],
                                              doc_text=wiki_text,
                                              ent_type=entity_dict["type"],
                                              facts=fact_dict,
                                              ))

    if add_nil:
        ontology.add(id="NIL",
                     oe=Ontology_Entity(id="NIL",
                                        name="Null entity",
                                        doc_text="",
                                        ent_type="",
                                        facts=[],
                                        ))
    return ontology


def main():
    res = load_ontology()

if __name__ == "__main__":
    main()
