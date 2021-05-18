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
from data.candidate_selection import CandidateSelection
import csv
import pickle
import traceback
import numpy as np
lang_dict = {"SPA": "es_core_news_md","CMN": "zh",  "ENG": "en_core_web_md", }


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

def fix_span(doc, doc_start, doc_end, log, q, dio):
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
            log.error(q.mention_string.iloc[0])
            log.error("|" + doc.text[doc_start:doc_end + 1] + "|")
            log.error(doc.text[doc_start - 10:doc_end + 10])
            raise Exception("Mention span not aligned: {0}".format(dio))
    return span

def load_data_split(args,partition="training", hold_out = 0.2, language = None):
    if language is None:
        language = args.language

    mentions, ontology = load_cached_data(language, args, partition)

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

def generate_all(args, partition="training", hold_out = 0.2):
    if not args.evaluation:
        if args.language.upper() == "ALL":
            for lang in lang_dict:
                lang_heldout, lang_ontology_eval = load_cached_data(lang, args, partition)

        else:
            lang_training, lang_heldout, lang_ontology = \
                load_data_split(args, partition, hold_out, args.language)
    else:
        if args.language.upper() == "ALL":
            for lang in lang_dict:
                lang_heldout, lang_ontology_eval = load_cached_data(lang, args, "eval")

        else:
            lang_heldout, lang_ontology_eval = load_cached_data(args.language, args, "eval")

        #lang_training, lang_ontology = load_data(lang, args, "training")
        #lang_ontology.entities.update(lang_ontology_eval.entities)

def load_data_split_all(args, partition="training", hold_out = 0.2):
    train_mentions_all = Mentions()
    heldout_mentions_all = Mentions()
    ontology_all = Ontology()
    holdout_languages = set(x for x in args.holdout_language.split(","))

    for lang in lang_dict:
        if not args.evaluation:
            lang_training, lang_heldout, lang_ontology = \
                load_data_split(args, partition, hold_out, lang)
        else:
            lang_training = Mentions()

            lang_heldout, lang_ontology = load_cached_data(lang, args, "eval")
            if lang not in holdout_languages:
                all_lang_training, lang_ontology_train = load_cached_data(lang, args, "training")
                for m in all_lang_training.entities:
                    if not m.nil:
                        lang_training.entities.append(m)
                lang_ontology.entities.update(lang_ontology_train.entities)
        ontology_all.entities.update(lang_ontology.entities)

        if lang not in holdout_languages:
            these_entities = []
            for m in lang_training.entities:
                these_entities.append(m)
            if args.training_ds.lower() == "random" and args.training_perc < 1.0:
                split_point = int(len(these_entities) * float(args.training_perc))
                these_entities = these_entities[:split_point]

                logging.getLogger().info("Limiting training for {0} to {1}".format(lang, len(these_entities)))

            elif len(args.training_ds) > 0:
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
                elif args.training_ds.lower() == "eq":
                    with open(f"{args.data_directory}/query_splits/training_query_ids.pkl", 'rb') as f:
                        training_query_ids = pickle.load(f)
                    these_entities = [x for x in these_entities if x.id in training_query_ids]


            if args.exclude_test_entities:
                heldout_entities = set(m.kbid for m in heldout_mentions_all.entities)
                these_entities = [x for x in these_entities if x.kbid not in heldout_entities]
                logging.getLogger().info(f"Removing all entities len {len(heldout_entities)} occurring in heldout, "
                                         f"leaving N= {len(these_entities)} training")

            with open(os.path.join(args.directory, "training_{0}.pkl".format(lang)), 'wb') as training_f:
                entity_ids = [x.id for x in these_entities]
                pickle.dump(obj=entity_ids, file=training_f)
            train_mentions_all.entities.extend(these_entities)


        else:
            logging.getLogger().info("Holding out {0} training".format(lang))

        if args.heldout_max_kb is not None:
            max_kb_int = int(args.heldout_max_kb)
            heldout_entities = []
            for m in lang_heldout.entities:
                heldout_entities.append(m)
            dataframe_dict = []
            for m in heldout_entities:
                dataframe_dict.append({"id": m.id, "kbid": m.kbid})
            heldout_entity_df = pd.DataFrame.from_dict(dataframe_dict)

            entity_df = heldout_entity_df.sample(frac=1).groupby('kbid').head(max_kb_int).reset_index(drop=True)
            logging.getLogger().info(set(entity_df['id']))
            heldout_entities = [x for x in heldout_entities if x.id in set(entity_df['id'])]

            logging.getLogger().info("Sampling max KB training for {0} to {1} with n={2}"
                                     .format(lang, len(heldout_entities), max_kb_int))
            heldout_mentions_all.entities.extend(heldout_entities)
        else:
            heldout_entities = []
            for m in lang_heldout.entities:
                heldout_entities.append(m)
            if args.training_ds.lower() == "eq":
                with open(f"{args.data_directory}/query_splits/eval_query_ids.pkl", 'rb') as f:
                    eval_query_ids = pickle.load(f)
                heldout_entities = [x for x in heldout_entities if x.id in eval_query_ids]

            for m in heldout_entities:
                heldout_mentions_all.entities.append(m)
        logging.getLogger().info(f"popularity {args.popularity}, {args.popularity_type}")

        if args.popularity and args.popularity_type == "all":
            pop_count = 0
            for m in lang_training.entities:
                if m.kbid in ontology_all.entities:
                    ontology_all.entities[m.kbid].popularity += 1.
                    pop_count += 1.
            logging.getLogger().info(f"Popularity type {args.popularity_type}, {pop_count}")
    logging.getLogger().info(f"popularity {args.popularity}, {args.popularity_type}")

    if args.match_train_lang is not None:
        match_training, _ = load_cached_data(
            args.match_train_lang, args, "training"
        )
        match_entities = sum(1 for x in match_training.entities if not x.nil)
        min_entities = min(match_entities, len(train_mentions_all.entities))

        logging.info(
            f"Matching training size to min {min_entities}:{len(train_mentions_all.entities)} to {args.match_train_lang}:{match_entities}"
        )

        random.Random(1).shuffle(train_mentions_all.entities)
        train_mentions_all.entities = train_mentions_all.entities[:min_entities]
        logging.info(f"training reduced to {len(train_mentions_all.entities)}")

    if args.popularity and args.popularity_type == "train":
        pop_count = 0
        for m in train_mentions_all.entities:
            ontology_all.entities[m.kbid].popularity += 1.
            pop_count += 1.
        logging.getLogger().info(f"Popularity type {args.popularity_type}, {pop_count}")
    return train_mentions_all, heldout_mentions_all, ontology_all

def load_cached_data(language, args, partition):
    cached_file = os.path.join(args.data_directory, language, f"{language}_{partition}.pkl")
    if os.path.exists(cached_file):
        with open(cached_file, 'rb') as cache:
            cached_data = pickle.load(cache)
            mentions = cached_data["mentions"]
            ontology = cached_data["ontology"]
            logging.getLogger().info(f"loaded {language} {partition} from {cached_file}")
    else:
        mentions, ontology = load_data(language, args, partition)
        with open(cached_file, 'wb') as cache:

            pickle.dump({
                "mentions" : mentions,
                "ontology" : ontology,
            }, cache)
    if args.training_ids and partition == "training":
        id_list = args.training_ids.replace("\"", "").split(",")
        for pair in id_list:
            pair_lang, pair_pkl = pair.split(":")
            if pair_lang == language:
                with open(pair_pkl, 'rb') as cache:
                    cached_data = pickle.load(cache)
                    ext_mentions = [x.id for x in cached_data["mentions"].entities]
                    mentions.entities = [x for x in mentions.entities if x.id in ext_mentions]

    if args.oracle:
        logging.getLogger().info(f"{language} {partition} set to oracle mode")
        for men in mentions.entities:
            if men.kbid not in men.candidate_kb and not men.nil:
                if len(men.candidate_kb) >= args.n_cands:
                    lowest_score = sorted(
                        men.candidate_kb.items(), key=lambda x: x[1], reverse=False
                    )[0]
                    del men.candidate_kb[lowest_score[0]]
                men.add_candidate(men.kbid, 234234.0)
    return mentions, ontology

def load_data(language, args,partition="training"):
    log =logging.getLogger()
    log.info(f"Processing {language} {partition}")
    data_directory = args.data_directory

    if language not in lang_dict.keys():
        raise NotImplementedError("Module accepts {0} only".format(",".join(lang_dict.keys())))

    try:
        spacy_model = spacy.load(lang_dict[language],  disable=["tagger", "ner"])
    except:
        spacy_model = spacy.blank(lang_dict[language])
        log.warning("Using blank spacy model")

    if language == "CMN":
        from spacy.pipeline import Sentencizer
        sentencizer = Sentencizer(punct_chars=[".", "?", "!", "。", ">", "<", "\n"])
        spacy_model.add_pipe(sentencizer)

    fbt.configure(home=args.fbt_path, config='config.dat')
    fbi = fbt.FreebaseIndex()
    fbi.describe()

    suffixes = spacy_model.Defaults.suffixes + (r'''-"@''',)
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    spacy_model.tokenizer.suffix_search = suffix_regex.search

    prefixes = spacy_model.Defaults.prefixes + (r'''-"@''',)
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    spacy_model.tokenizer.prefix_search = prefix_regex.search

    query_filename = os.path.join(data_directory, language, f"tac_kbp_2015_tedl_{partition}_gold_standard_entity_mentions.tab")
    header_list = ["system_id", "query_id", "mention_string", "doc_id_offsets", "link_id", "entity_type", "mention_type",
                   "confidence", "web_search", "wiki_text", "unknown"]
    queries = pd.read_csv(query_filename, iterator=True, sep='\t', names=header_list, header=None, chunksize=1)
    included_links = set()
    mentions = Mentions()
    counter_dict = defaultdict(lambda : 0)
    ontology = Ontology()
    no_name_set = set()
    no_name_count = 0
    mention_csv_file = open(os.path.join(args.directory, f"mentions_{language}_{partition}.csv"), 'w')
    mention_csv = csv.DictWriter(mention_csv_file, fieldnames=["mention", "id", "doc_id"])
    mention_csv.writeheader()
    mention_csv_cands = []
    if partition == "training":
        mention_candidates = pd.read_csv(os.path.join(data_directory, language, "mentions_{0}_out.csv".format(language)))
    else:
        mention_candidates = pd.read_csv(os.path.join(data_directory, language, "mentions_{0}_out_eval.csv".format(language)))

    cand_sel = CandidateSelection(args)
    wiki_titles = defaultdict(lambda: 0)
    type_count = defaultdict(lambda: 0)
    field_counts = defaultdict(lambda: 0)
    no_type = 0
    typed = 0
    non_wiki_nils = 0
    no_context = 0
    gold_cand_included = 0
    if partition == "training":
        candidate_pkl_filename = os.path.join(data_directory, language, "candidates_{index}_{k}.pkl"
                                              .format(index=os.path.basename(fbi.index), k=args.n_cands))
    else:
        candidate_pkl_filename = os.path.join(data_directory, language, f"candidates_{os.path.basename(fbi.index)}_{args.n_cands}_{partition}.pkl")

    candidate_dict = {}
    load_candidates = True
    # if os.path.exists(candidate_pkl_filename) and not args.reload_cands:
    #     with open(candidate_pkl_filename, 'rb') as pkl_f:
    #         candidate_dict = pickle.load(pkl_f)
    #         load_candidates = False
    #         log.info("Loading candidates from {0}".format(candidate_pkl_filename))

    for r, q in enumerate(queries):
        dio = re.match(r"([^:]*):([0-9]+)-([0-9]+)", q.doc_id_offsets.iloc[0])
        doc_id = dio.group(1)
        doc_start = int(dio.group(2))
        doc_end = int(dio.group(3))
        doc_language, _ = doc_id.split("_", 1)
        if doc_language == language:
            if args.test_exclude_nils and not q["link_id"].iloc[0].startswith("m."):
                continue
            try:
                document_filename = next(Path(os.path.join(data_directory, language, partition)).rglob(doc_id+"*"))
            except:
                raise Exception("Error locating {0}, id {1}".format(Path(os.path.join(data_directory, language, partition)), doc_id ))



            with open(document_filename, 'r', encoding='utf8') as doc_f:
                doc = spacy_model(doc_f.read().replace("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n", "", 1))
            span = doc.char_span(doc_start, doc_end+1)
            if not span:
                span = fix_span(doc, doc_start, doc_end, log, q, dio)
                if not span:
                    log.error(q)
                    log.error(q["link_id"])
                    raise Exception("Span not found for {0}".format(q["link_id"]))
            kbid = q["link_id"].iloc[0]
            if "NIL" in kbid:
                kbid = "NIL"
            ent = Mention_Entity(id=q["query_id"].iloc[0], mention=span, doc = doc, kbid=kbid,kb_type=q["entity_type"].iloc[0],
                                 doc_filename=document_filename, lang=language, doc_id = doc_id)
            if span.sent is None:
                print(q)
            ent.nil = "NIL" in ent.kbid
            if q["link_id"].iloc[0].startswith("m."):
                included_links.add(q["link_id"].iloc[0])
                counter_dict["LINK"] += 1
            else:
                counter_dict["NULL"] += 1
                ent.nil = True

            link_id = q["link_id"].iloc[0].replace("\n", "")
            no_name_flag = False
            if q["link_id"].iloc[0].startswith("m.") and link_id not in ontology.entities:

                if link_id not in no_name_set:
                    link_info = fbi.fetch("f_{0}".format(link_id))
                    added = False
                    try:
                        if len(link_info) > 0:
                            added = ontology.add(link_id, link_info, add_if_no_name=True)
                        else:
                            link_info = {'subject':"f_{0}".format(link_id)}
                            added = ontology.add(link_id, link_info, add_if_no_name=True)

                    except Exception as e:
                        log.info(e)
                        log.info(traceback.format_exc())
                        log.info("Error!")
                        log.info(link_id)
                        log.info(link_info)

                    if not added:
                        no_name_set.add(link_id)
                        no_name_flag = True
                    elif args.exclude_nonwiki and not \
                            any(x for x in ontology.entities[link_id].fields if 'fk_key.wikipedia' in x):
                        del ontology.entities[link_id]
                        ent.nil = True
                        non_wiki_nils += 1
            if link_id in ontology.entities and ontology.get_text(ontology.entities[link_id]).startswith("f_m"):
                no_context += 1

            if q["link_id"].iloc[0].startswith("m.") and no_name_flag:
                no_name_count += 1
                ent.nil = True
                ent.kbid = "NIL"
            if link_id in ontology.entities or (ent.nil and not args.test_exclude_nils):
                if load_candidates:
                    try:
                        ent, ontology, cand_titles, ent_cached = cand_sel.pull_candidates(ent, mention_candidates, fbi,
                                                                                ontology, args,
                                                                                n_cands=args.n_cands,
                                                                                exclude_nonwikis=args.exclude_nonwiki)
                        candidate_dict[ent.id] = ent_cached
                    except Exception as e:
                        log.info(q["link_id"].iloc[0])
                        log.info(q["query_id"].iloc[0])
                        log.info(ent)
                        log.error(str(e))
                        log.error(traceback.format_exc())
                else:
                    try:
                        ent, ontology = cand_sel.pull_candidates_cached(ent, candidate_dict, ontology,
                                                               n_cands=args.n_cands,
                                                               exclude_nonwikis=args.exclude_nonwiki)
                    except Exception as e:
                        log.info(q["link_id"].iloc[0])
                        log.info(q["query_id"].iloc[0])
                        log.error(str(e))
                        log.error(traceback.format_exc())

                        log.info(ent)
                mentions.entities.append(ent)
                if ent.kbid in ent.candidate_kb:
                    gold_cand_included += 1
                    # if cand_sel.query_tracing[ent.id]["source"] == "NA":
                    #     print(ent)
            else:
                log.info("EXCLUSION")
                log.info(q["link_id"].iloc[0])
                log.info(q["query_id"].iloc[0])
                log.info(ent)

            if link_id in ontology.entities and "fk_key.wikipedia.en_title" in ontology.entities[link_id].fields:
                wiki_titles[ontology.entities[link_id].fields["fk_key.wikipedia.en_title"]] += 1

            mention_csv.writerow({"mention": ent.mention, "id": ent.id, "doc_id": ent.doc_filename})

            if r % 500 == 0:
                log.info(f"Processed {r}")
    mention_csv_file.close()
    cand_sel.tracing_results(args, language, partition)
    dataframe = pd.DataFrame.from_dict(mention_csv_cands)
    dataframe.to_csv(os.path.join(args.directory, "mentions_cands_{0}.csv".format(language)), index=False)
    log.info("No context for gold entity: {0}".format(no_context / float(len(mentions.entities))))
    # if load_candidates:
    #     with open(candidate_pkl_filename, 'wb') as pkl_f:
    #
    #         pickle.dump(candidate_dict, pkl_f)
    #         log.info("Saving candidates to {0}".format(candidate_pkl_filename))

    with open(os.path.join(args.directory, "wiki_titles.txt"), 'w') as wf:
        for wt in wiki_titles:
            wf.write("{0}\t{1}\n".format(wt, wiki_titles[wt]))
    log.info("Loaded {0} mentions".format(sum(v for x,v in counter_dict.items())))
    log.info("{0} with links, {1} null".format(counter_dict["LINK"], counter_dict["NULL"]))
    log.info("{0} with no names, {1} unique".format(no_name_count, len(no_name_set)))
    log.info("Total mentions:{0}".format(len(mentions.entities)))
    log.info("Gold candidates included for: {0}, {1}%".format(gold_cand_included, gold_cand_included /
                                                              sum(1. for x in mentions.entities if not x.nil)))
    log.info(f"Mean candidate size {np.mean([len(x.candidate_kb) for x in mentions.entities if not x.nil])} ")
    log.info(f"{ sum(1. for x in mentions.entities if x.nil)} NILs, "
             f"with {np.mean([len(x.candidate_kb) for x in mentions.entities if x.nil])} mean candidates")
    log.info("Non-wiki ents set to nil:{0}".format(non_wiki_nils))

    log.info(f"nil skipped candgen for {sum(1. for x in mentions.entities if x.cand_skip and x.nil)} nils, "
             f"{sum(1. for x in mentions.entities if x.cand_skip and x.nil) / sum(1. for x in mentions.entities if x.nil)} perc")

    log.info(f"non nil skipped candgen for {sum(1. for x in mentions.entities if x.cand_skip and not x.nil)} nils, "
             f"{sum(1. for x in mentions.entities if x.cand_skip and not x.nil) / sum(1. for x in mentions.entities if not x.nil)} perc")

    for lang_id in ["zh", "es"]:
        name_found = 0.
        text_found = 0.
        total = 0.
        for e in mentions.entities:
            if not e.nil and lang_id in ontology.entities[e.kbid].l2_names:
                name_found += 1.
            if not e.nil and lang_id in ontology.entities[e.kbid].l2_text:
                text_found += 1.
            total += 1.
        log.info(f"{lang_id} {name_found / total} names included for gold candidates")
        log.info(f"{lang_id} {text_found / total} text included for gold candidates")

        name_found = 0.
        text_found = 0.

        total = 0.
        for e in ontology.entities.values():
            if lang_id in e.l2_names:
                name_found += 1.
            if lang_id in e.l2_text:
                text_found += 1.
            total += 1.
        log.info(f"{lang_id} {name_found / total} names included for ontology")
        log.info(f"{lang_id} {text_found / total} text included for ontology")


    """ 
    
    log.info("No type:{0}, typed:{1}".format(no_type, typed))
    type_list = sorted(type_count.items(), key=lambda x: x[1], reverse=True)
    for x,y in type_list:
        log.info("Type:{0}:{1}".format(x, y))

    field_list = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
    for x,y in field_list:
        log.info("Field:{0}:{1}".format(x, y))
    """
    return mentions, ontology


def main():
    res = load_data(exclude_nils=True)


if __name__ == "__main__":
    main()
