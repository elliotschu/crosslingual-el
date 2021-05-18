"""
Elliot Schumacher, Johns Hopkins University
Created 6/19/18
"""
import numpy as np
from concrete.util import file_io
import argparse
import os
import pickle
from gensim.models import KeyedVectors

def main():
    datadir = "/Users/elliotschumacher/Dropbox/git/synonym_detection/resources/bilm/standard" #os.path.join('tests', 'fixtures', 'model')

    concrete_zip = "/Users/elliotschumacher/Dropbox/concept/share_clef/SPLIT_2017-12-08-13-38-01/train/train_concrete.zip"
    #concrete_zip = "/Users/elliotschumacher/Dropbox/concept/share_clef/SPLIT_2017-12-08-13-38-01/train/train_concrete.zip"

    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v_file', default="/Users/elliotschumacher/Dropbox/git/concept-linker/w2v/mimic_w2v_600n_0")
    parser.add_argument('--concrete_file', default=concrete_zip)
    parser.add_argument('--concrete_file2',
                        default="/Users/elliotschumacher/Dropbox/concept/share_clef/SPLIT_2017-12-08-13-38-01/test/test_concrete.tar")
    parser.add_argument('--output_dir', default=os.path.join(datadir, 'mention_embeddings_w2v_test'))

    args = parser.parse_args()

    concrete_zip = args.concrete_file
    os.makedirs(args.output_dir, exist_ok=True)

    comm_dict = {filename: comm for (comm, filename) in file_io.CommunicationReader(concrete_zip)}

    if args.concrete_file2 is not None:
        comm_dict2 = {filename: comm for (comm, filename) in file_io.CommunicationReader(args.concrete_file2)}
        comm_dict.update(comm_dict2)
        print("Adding concrete file {0}".format(args.concrete_file2))

    mention_to_info = {}
    id_to_mention_info = {}
    indx = 0

    num_lines = sum(len(comm.entityMentionSetList[0].mentionList) for _, comm in comm_dict.items())
    max_len = 0
    for _, comm in comm_dict.items():
        this_max = max(len(men.tokens.tokenIndexList) for men in comm.entityMentionSetList[0].mentionList)
        max_len = max(this_max, max_len)
    w2v = KeyedVectors.load_word2vec_format(args.w2v_file, binary=True)
    mention_representation = np.zeros((num_lines, w2v.vector_size))


    sentence_id = 0
    #with open(dataset_file, 'r') as fin, h5py.File(outfile, 'w') as fout:
    #    for line in fin:
    for c_i, (comm_file, comm) in enumerate(comm_dict.items()):

        for men in comm.entityMentionSetList[0].mentionList:
            if men.tokens.tokenization is None:
                print(men.uuid.uuidString)
            else:
                [w.text.strip() for w in men.tokens.tokenization.tokenList.tokenList]

                word_vects = []
                for m_i in men.tokens.tokenIndexList:
                    word = men.tokens.tokenization.tokenList.tokenList[m_i].text.strip()
                    try:
                        word_vects.append(w2v[word])
                    except:
                        print("Missing word :{0}".format(word))
                if len(word_vects) == 0:
                    word_vects.append(np.zeros(w2v.vector_size))

                embeddings = np.stack(word_vects)
                men_max = np.max(embeddings, axis=0)

                mention_map = {"comm_uuid": comm.uuid.uuidString,
                               "mention_uuid": men.uuid.uuidString,
                               "index": indx
                               }

                mention_to_info[men.uuid.uuidString] = mention_map
                id_to_mention_info[indx] = mention_map
                mention_representation[indx,:] = men_max
                indx += 1
                sentence_id += 1
        print("Completed comm {0}".format(comm_file))
    with open(os.path.join(args.output_dir, 'mention_to_info.pkl'), 'wb') as out_dict:
        pickle.dump(mention_to_info, out_dict)

    with open(os.path.join(args.output_dir, 'id_to_mention_info.pkl'), 'wb') as out_dict:
        pickle.dump(id_to_mention_info, out_dict)

    with open(os.path.join(args.output_dir, 'mention_representations.npy'), 'wb') as out_dict:
        np.save(out_dict, mention_representation)

if __name__ == "__main__":
    main()