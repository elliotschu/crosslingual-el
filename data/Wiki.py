"""
Elliot Schumacher, Johns Hopkins University
Created 10/14/19
"""
import spacy
import pandas as pd
import random
import logging
from pathlib import Path
import os
from collections import defaultdict
import textdistance
import basekb.fbtools as fbt
from data.Objects import *
import csv
import pickle
import traceback

lang_dict = {"ru" : "ru",
             "ar" : "ar",
             "ko" : "ko",
             "fa" : "fa",}


def adjust_span(doc, doc_start, doc_end):
    with doc.retokenize() as retokenizer:
        for tok in doc:
            if doc_start in range(tok.idx, tok.idx + len(tok)) and doc_end in range(tok.idx, tok.idx + len(tok)):
                new_tokens = []
                if tok.idx < doc_start:
                    new_tokens.append(doc.text[tok.idx:doc_start])
                new_tokens.append(doc.text[doc_start:doc_end + 1])
                if doc_end + 1 < (tok.idx + len(tok)):
                    new_tokens.append(doc.text[doc_end + 1:tok.idx + len(tok)])

                attrs = {"POS": [tok.pos for _ in new_tokens], "DEP": [tok.dep for _ in new_tokens]}
                heads = [tok.head for _ in new_tokens]
                retokenizer.split(tok, new_tokens, heads=heads, attrs=attrs)
                break
            elif doc_start in range(tok.idx, tok.idx + len(tok)):
                new_tokens = []
                if tok.idx < doc_start:
                    new_tokens.append(doc.text[tok.idx:doc_start])
                new_tokens.append(doc.text[doc_start:tok.idx + len(tok)])

                attrs = {"POS": [tok.pos for _ in new_tokens], "DEP": [tok.dep for _ in new_tokens]}
                heads = [tok.head for _ in new_tokens]
                retokenizer.split(tok, new_tokens, heads=heads, attrs=attrs)
            elif doc_end in range(tok.idx, tok.idx + len(tok)):
                new_tokens = []
                new_tokens.append(doc.text[tok.idx:doc_end + 1])
                if doc_end + 1 < (tok.idx + len(tok)):
                    new_tokens.append(doc.text[doc_end + 1:tok.idx + len(tok)])

                attrs = {"POS": [tok.pos for _ in new_tokens], "DEP": [tok.dep for _ in new_tokens]}
                heads = [tok.head for _ in new_tokens]
                retokenizer.split(tok, new_tokens, heads=heads, attrs=attrs)
                break

def fix_span(doc, doc_start, doc_end, log, ment):
    for index_pos in range(doc_start, doc_end):
        if doc.text[index_pos:index_pos + 1].isspace():
            doc_start = doc_start + 1
        else:
            break
    for index_pos in range(doc_end + 1, doc_start, -1):
        if doc.text[index_pos - 1:index_pos].isspace():
            doc_end = doc_end - 1
        else:
            break
    span = doc.char_span(doc_start, doc_end + 1)
    if not span:
        adjust_span(doc, doc_start, doc_end)
        span = doc.char_span(doc_start, doc_end + 1)

        if not span:
            log.error("Mention span not aligned")
            log.error("|" + doc.text[doc_start:doc_end + 1] + "|")
            log.error(doc.text[doc_start - 10:doc_end + 10])
            raise Exception("Mention span not aligned: {0}".format(ment))
    return span

def load_data_split(args,partition="training", hold_out = 0.2, language = None):
    if language is None:
        language = args.language

    mentions, ontology = load_data(language, args, partition)

    file_list = list(set(e.doc_filename for e in mentions.entities))
    file_list.sort()
    random.Random(1).shuffle(file_list)
    train = file_list[:int(len(file_list) * (1 - hold_out))]
    holdout = file_list[int(len(file_list) * (1 - hold_out)):]
    logging.getLogger().info(holdout)

    train_mentions = Mentions()
    for m in mentions.entities:
        if m.doc_filename in train and not m.nil:
            train_mentions.entities.append(m)
    holdout_mentions = Mentions()
    logging.getLogger().info("Heldout files: {0}".format(",".join([str(os.path.basename(x)) for x in holdout])))
    heldout_wikiperc = 0
    heldout_nocands = 0
    for m in mentions.entities:
        if m.doc_filename in holdout:
            if args.test_exclude_nils and not m.nil:
                holdout_mentions.entities.append(m)
            elif not args.test_exclude_nils:
                holdout_mentions.entities.append(m)

            if m.kbid not in m.candidate_kb:
                heldout_nocands += 1
            elif any(x for x in ontology.entities[m.kbid].fields if 'fk_key.wikipedia' in x):
                heldout_wikiperc += 1
    logging.getLogger().info("Heldout wiki perc: {0}".format(heldout_wikiperc / float(len(holdout_mentions.entities))))
    logging.getLogger().info("Heldout not cand perc: {0}".format(heldout_nocands / float(len(holdout_mentions.entities))))

    return train_mentions, holdout_mentions, ontology

def load_data_split_all(args, partition="training", hold_out = 0.2):
    train_mentions_all = Mentions()
    heldout_mentions_all = Mentions()
    ontology_all = Ontology()
    holdout_languages = set(x for x in args.holdout_language.split(","))

    for lang in lang_dict:
        lang_training, lang_heldout, lang_ontology = \
            load_data_split(args, partition, hold_out, lang)
        ontology_all.entities.update(lang_ontology.entities)
        if lang not in holdout_languages:
            these_entities = []
            for m in lang_training.entities:
                these_entities.append(m)
            if args.training_ds.lower() == "random":
                split_point = int(len(these_entities) * float(args.training_perc))
                these_entities = these_entities[:split_point]

                logging.getLogger().info("Limiting training for {0} to {1}".format(lang, len(these_entities)))

            else:
                dataframe_dict = []
                for m in these_entities:
                    dataframe_dict.append({"id" : m.id, "kbid" : m.kbid})
                entity_df = pd.DataFrame.from_dict(dataframe_dict)
                if args.training_ds.lower() == "max_kb":
                    max_kb_int = int(args.max_kb)

                    entity_df = entity_df.sample(frac=1).groupby('kbid').head(max_kb_int).reset_index(drop=True)
                    logging.getLogger().info(set(entity_df['id']))
                    these_entities = [x for x in these_entities if x.id in set(entity_df['id'])]
                    logging.getLogger().info("Sampling max KB training for {0} to {1} with n={2}"
                                             .format(lang, len(these_entities), max_kb_int))
                elif args.training_ds.lower() == "head":
                    entity_df['link_count'] = entity_df['kbid'].map(entity_df['kbid'].value_counts())
                    entity_df = entity_df.sort_values("link_count", ascending=False)
                    entity_df = entity_df.head(int(len(entity_df) * float(args.training_perc)))
                    these_entities = [x for x in these_entities if x.id in set(entity_df['id'])]
                    logging.getLogger().info("Head sampling for {0} to {1} with perc={2}"
                                             .format(lang, len(these_entities), args.training_perc))
                elif args.training_ds.lower() == "tail":
                    entity_df['link_count'] = entity_df['kbid'].map(entity_df['kbid'].value_counts())
                    entity_df = entity_df.sort_values("link_count", ascending=False)
                    entity_df = entity_df.tail(int(len(entity_df) * float(args.training_perc)))
                    these_entities = [x for x in these_entities if x.id in set(entity_df['id'])]
                    logging.getLogger().info("Tail sampling for {0} to {1} with perc={2}"
                                             .format(lang, len(these_entities), args.training_perc))
            with open(os.path.join(args.directory, "training_{0}.pkl".format(lang)), 'wb') as training_f:
                entity_ids = [x.id for x in these_entities]
                pickle.dump(obj=entity_ids, file=training_f)
            train_mentions_all.entities.extend(these_entities)
            logging.getLogger().info(f"Training {lang}, entities {len(these_entities)}")
            if args.popularity and args.popularity_type == "all":
                pop_count = 0
                for m in lang_training.entities:
                    if m.kbid in ontology_all.entities:
                        ontology_all.entities[m.kbid].popularity += 1
                        pop_count += 1
                logging.getLogger().info(f"Popularity type {args.popularity_type}, {pop_count}")
        else:
            logging.getLogger().info("Holding out {0} training".format(lang))
        for m in lang_heldout.entities:
            heldout_mentions_all.entities.append(m)
        logging.getLogger().info(f"Heldout {lang}, entities {len(lang_heldout.entities)}")


    if args.popularity and args.popularity_type == "train":
        pop_count = 0
        for m in train_mentions_all.entities:
            ontology_all.entities[m.kbid].popularity += 1
            pop_count += 1
        logging.getLogger().info(f"Popularity type {args.popularity_type}, {pop_count}")
    logging.getLogger().info(f"Training data {len(train_mentions_all.entities)}")
    logging.getLogger().info(f"Heldout data {len(heldout_mentions_all.entities)}")
    logging.getLogger().info(f"Ontology data {len(ontology_all.entities)}")
    return train_mentions_all, heldout_mentions_all, ontology_all

def pull_candidates_cached(ent, cached_candidates, ontology,sent_id, link_id, candidate_pages, n_cands = 200):
    these_cands = cached_candidates.loc[(cached_candidates['sent_id'] == sent_id) &
                                        (cached_candidates['m_id'].astype(str) == link_id)]

    added_count = 0

    for i, candidate in these_cands.iterrows():
        if candidate["cand_title"] != 'NULLTITLE':
            try:
                if added_count >= n_cands:
                    break
                candidate_pg = candidate_pages[candidate["cand_title"]]#[v for v in candidate_pages.values() if v['title'] == candidate["cand_title"]][0]
                ontology.add_wiki(candidate_pg.get("_id"), candidate_pg, add_if_no_name=False)
                ent.add_candidate(candidate_pg.get("_id"), candidate['cand_score_ts'])
            except:
                pass # logging.getLogger().warning(f"No page found for {candidate['cand_title']}")

    return ent, ontology

def load_data(language, args,partition="training"):
    log =logging.getLogger()

    data_directory = args.data_directory

    if language not in lang_dict.keys():
        raise NotImplementedError("Module accepts {0} only".format(",".join(lang_dict.keys())))

    try:
        spacy_model = spacy.load(lang_dict[language])
    except:
        spacy_model = spacy.blank(lang_dict[language])
        log.warning("Using blank spacy model")

    from spacy.pipeline import Sentencizer
    sentencizer = Sentencizer(punct_chars=[".", "?", "!", "。", ">", "<", "\n"])
    spacy_model.add_pipe(sentencizer)

    # load wiki data
    with open(os.path.join(args.data_directory, language, f'wiki_{language}.pkl'), 'rb') as f:
        wiki_info = pickle.load(f)
        en_pages = wiki_info["en_pages"]
        documents = wiki_info["mentions"][language]
        nil_documents = wiki_info["nil_mentions"][language]
        l2_pages = wiki_info["l2_pages"][language]

    with open(os.path.join(args.data_directory, language, f'wiki.{language}.cands.pkl'), 'rb') as f:
        candidate_pages = pickle.load(f)

    mention_candidates = pd.read_csv(os.path.join(data_directory, language, f'mention.{language}.cands.csv'))
    mentions = Mentions()
    included_links = set()
    gold_cand_included = 0
    counter_dict = defaultdict(lambda : 0)
    ontology = Ontology()

    for doc_title in documents:
        article_dict = l2_pages[doc_title]
        doc = spacy_model(article_dict['article'])
        for (sent_id, link_id), ment_dict in documents[doc_title].items():
            sentence = ment_dict['sent']
            link = ment_dict['link']
            doc_start = int(link['start'])
            doc_end = int(link['end'])

            sent_start = int(sentence['start'])
            span = doc.char_span(doc_start+sent_start,  doc_end+sent_start)
            if not span:
                span = fix_span(doc, doc_start, doc_end, log, ment_dict)
                if not span:
                    log.error(ment_dict)
                    raise Exception("Span not found for {0}".format(f'{sent_id}_{link_id}'))
            kbid = link['id_ll']

            ent = Mention_Entity(id=f'{sent_id}_{link_id}_{language}', doc=doc, mention=span, kbid=kbid,
                                 kb_type=None, doc_filename=sentence['title'], lang=language, doc_id=sentence['title'])
            if span.sent is None:
                log.warning(link)
            included_links.add(kbid)
            counter_dict["LINK"] += 1
            if kbid not in ontology.entities:
                ontology.add_wiki(kbid, en_pages[link['title_ll']], add_if_no_name=True)

            ent, ontology = pull_candidates_cached(ent, mention_candidates, ontology,sent_id, link_id, candidate_pages, n_cands=args.n_cands)

            mentions.entities.append(ent)
            if ent.kbid in ent.candidate_kb:
                gold_cand_included += 1

    for doc_title in nil_documents:
        article_dict = l2_pages[doc_title]
        doc = spacy_model(article_dict['article'])
        for (sent_id, link_id), ment_dict in nil_documents[doc_title].items():
            sentence = ment_dict['sent']
            link = ment_dict['link']
            doc_start = int(link['start'])
            doc_end = int(link['end'])

            sent_start = int(sentence['start'])
            span = doc.char_span(doc_start+sent_start,  doc_end+sent_start)
            if not span:
                span = fix_span(doc, doc_start, doc_end, log, ment_dict)
                if not span:
                    log.error(ment_dict)
                    raise Exception("Span not found for {0}".format(f'{sent_id}_{link_id}'))
            kbid = "NIL"

            ent = Mention_Entity(id=f'{sent_id}_{link_id}_{language}', doc=doc, mention=span, kbid=kbid,
                                 kb_type=None, doc_filename=sentence['title'], lang=language, doc_id=sentence['title'],)
            ent.nil = True
            if span.sent is None:
                log.warning(link)
            included_links.add(kbid)
            counter_dict["LINK"] += 1
            # if kbid not in ontology.entities:
            #     ontology.add_wiki(kbid, en_pages[link['title_ll']], add_if_no_name=True)

            ent, ontology = pull_candidates_cached(ent, mention_candidates, ontology,sent_id, link_id, candidate_pages, n_cands=args.n_cands)

            mentions.entities.append(ent)


    return mentions, ontology


def main():
    res = load_data(exclude_nils=True)


if __name__ == "__main__":
    main()
