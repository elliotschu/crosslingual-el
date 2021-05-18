"""
Elliot Schumacher, Johns Hopkins University
Created 6/19/18
"""
from concrete.util import file_io
import argparse
import os
import csv
from collections import defaultdict
import logging
import uuid
import pickle
import nltk
import numpy as np
import nltk.data
from gensim.models import KeyedVectors

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def main():
    datadir = "/Users/elliotschumacher/Dropbox/git/synonym_detection/resources/bilm/out_max" #os.path.join('tests', 'fixtures', 'model')

    concrete_zip = "/Users/elliotschumacher/Dropbox/git/concept-linker/lexicon.tsv"


    parser = argparse.ArgumentParser()
    parser.add_argument('--lexicon_file', default=concrete_zip)
    parser.add_argument('--w2v_file', default = "/Users/elliotschumacher/Dropbox/git/concept-linker/w2v/mimic_w2v_600n_0")

    parser.add_argument('--output_dir', default=os.path.join(datadir, 'embedding_output_w2v'))

    parser.add_argument('--include_alt', default=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    existing_list = list()

    concept_to_id_name_alt = {}
    id_to_concept_name_alt = {}
    indx = 0

    w2v = KeyedVectors.load_word2vec_format(args.w2v_file, binary=True)

    num_lines = 0
    with open(args.lexicon_file) as lex_file:
        tsv_reader = csv.reader(lex_file, delimiter="\t")
        for row in tsv_reader:
            num_lines += 1
            if args.include_alt and row[7].strip() != "":
                alt_names = set(row[7].split("|"))
                num_lines += len(alt_names)

    concept_representations = np.zeros((num_lines, w2v.vector_size))

    sentence_id = 0

    with open(args.lexicon_file) as lex_file:
        tsv_reader = csv.reader(lex_file, delimiter="\t")
        for row in tsv_reader:
            name = row[0]
            conceptId = row[1]
            #concept_group = fout.create_group(conceptId)

            concept_map = {"name": name,
                           "concept_id": conceptId,
                           "alternate": False,
                           "index": indx
                           }

            concept_to_id_name_alt[conceptId] = [concept_map]
            id_to_concept_name_alt[indx] = concept_map

            sentence = nltk.word_tokenize(name)
            word_vects = []
            for word in sentence:
                try:
                    word_vects.append(w2v[word])
                except:
                    print("Missing word :{0}".format(word))
            if len(word_vects) == 0:
                word_vects.append(np.zeros(w2v.vector_size))

            embeddings = np.stack(word_vects)

            men_max = np.max(embeddings, axis=0)
            out_max = men_max
            """
            ds3 = concept_group.create_dataset(
                indx, out_max.shape, dtype='float32',
                data=out_max
            )
            """
            concept_representations[indx, :] = out_max

            print("{concept}:{uuid}:{name}".format(concept=conceptId, uuid=indx, name=name))
            indx += 1

            if args.include_alt and row[7].strip() != "":
                alt_names = set(row[7].split("|"))
                for an in alt_names:
                    concept_map = {"name": an,
                                   "concept_id": conceptId,
                                   "alternate": True,
                                   "index": indx
                                   }
                    concept_to_id_name_alt[conceptId].append(concept_map)
                    id_to_concept_name_alt[indx] = concept_map

                    sentence = nltk.word_tokenize(an)
                    word_vects = []
                    for word in sentence:
                        try:
                            word_vects.append(w2v[word])
                        except:
                            print("Missing word :{0}".format(word))
                    if len(word_vects) == 0:
                        word_vects.append(np.zeros(w2v.vector_size))

                    embeddings = np.stack(word_vects)

                    men_max = np.max(embeddings, axis=0)
                    out_max = men_max
                    concept_representations[indx, :] = out_max
                    indx += 1
                    print("Alt {concept}:{uuid}:{name}".format(concept=conceptId, uuid=indx, name=an))

            sentence_id += 1

    concept_representations = concept_representations
    with open(os.path.join(args.output_dir, 'concept_to_id_name_alt.pkl'), 'wb') as out_dict:
        pickle.dump(concept_to_id_name_alt, out_dict)

    with open(os.path.join(args.output_dir, 'id_to_concept_name_alt.pkl'), 'wb') as out_dict:
        pickle.dump(id_to_concept_name_alt, out_dict)

    with open(os.path.join(args.output_dir, 'concept_representations.npy'), 'wb') as out_dict:
        np.save(out_dict, concept_representations)
    print("{0} duplicate concepts".format(len(existing_list)))
    for n, c in existing_list:
        print("{0}\t{1}".format(n, c))


if __name__ == "__main__":
    main()