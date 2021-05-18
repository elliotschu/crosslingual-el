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

def load_data(language = 'es',partition="train"):
    log =logging.getLogger()
    data_directory = "../datasets/mcnamee2011/"

    try:
        spacy_model = spacy.load(language)
    except:
        spacy_model = spacy.blank(language)
        log.warning("Using blank spacy model")

    query_filename = os.path.join(data_directory, "xlel-1.1/hand-curated-queries/{language}/PER-{partition}-queries".
                                    format(language=language, partition=partition))
    header_list = ["id", "string", "file", "entity_type", "kbid"]
    queries = pd.read_csv(query_filename, iterator=True, sep='\t', names=header_list, header=None, chunksize=1)
    mentions = Mentions()
    for q in queries:
        id = q["id"].iloc[0]

        document_filename = os.path.join(data_directory, subdirectories[language], language, language, q["file"].iloc[0].replace(".txt", ""))
        with open(document_filename, 'r', encoding='utf8') as doc_f:
            doc = spacy_model(doc_f.read())

        expression = r"{0}".format(q["string"].iloc[0].replace(" ", "\s*"))

        match_found = False
        for i, match in enumerate(re.finditer(expression, doc.text)):
            match_found = True
            start, end = match.span()
            span = doc.char_span(start, end)

            # This is a Span object or None if match doesn't map to valid token sequence

            if span:

                ent = Mention_Entity(id =id, doc =doc, mention=span, kbid= q["kbid"].iloc[0])

                mentions.entities.append(ent)
            else:
                log.error("Span not found for id {id}".format(id=id))

                with doc.retokenize() as retokenizer:
                    #find start token
                    for tok in doc:
                        if tok.start < start and tok.end > end:
                            heads = [(doc[3], 1), doc[2]]
                            attrs = {"POS": [tok.pos, tok.pos],
                                     "DEP": [tok.dep, tok.dep]}
                            retokenizer.split(doc[3], ["New", "York"], heads=heads, attrs=attrs)


        if id == 'Q-projsynd-PER-rev1-1118':
            print(q)

        """
        if i > 0:
            log.warning("For id {id}, {i} instances found".format(id = id, i = i+1))"""

        if not match_found:
            log.error(q)
            log.error("No match found")
            raise Exception("No match found")
    return mentions

def main():
    res = load_data()
    for r in res:
        print (r)

if __name__ == "__main__":
    main()
