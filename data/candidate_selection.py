"""
Elliot Schumacher, Johns Hopkins University
Created 3/6/20
"""
import re
import logging
import math
import os
import textdistance
from collections import defaultdict
import traceback
import pandas as pd

class CandidateSelection(object):
    def __init__(self, args):
        self.args = args
        self.cached_queries = {}
        self.query_tracing = {}
        self.exclude_types = set()
        if args.excluded_types:
            with open(args.excluded_types) as ex_f:
                for line in ex_f:
                    self.exclude_types.add(line.strip())
        self.jw = textdistance.JaroWinkler()


    def tracing_results(self, args, lang, partition):
        tracing_df = pd.DataFrame.from_dict([x for x in self.query_tracing.values()])
        tracing_df.to_csv(path_or_buf=os.path.join(args.directory, f"tracing{partition}{lang}.csv"))

    def trace_gold (self, candidates, ent, query_type, this_cand = None, this_cand_i = None):
        gold_found = False

        if ent.id not in self.query_tracing and not ent.nil:
            gold_ent = [x for x in candidates.values() if x['subject'].replace("f_", "") == f"{ent.kbid}"]
            if len(gold_ent) > 0:
                self.query_tracing[ent.id] = {
                    "source" : query_type,
                    "ent_id" : ent.id,
                    "mention_string" : ent.mention.text,
                    "cand" : "" if this_cand is None else this_cand['title'],
                    "cand_i" : this_cand_i,
                    "cand_score" : 0 if this_cand is None else this_cand['score_ts'],
                    "nil" : False,

                }
                gold_found = True
        else:
            gold_found = True
        return gold_found

    def trace_end(self, ent, these_cands, sorted_candidates, ontology):
        if ent.id not in self.query_tracing:
            self.query_tracing[ent.id] = {
                "source": "NA",
                "ent_id": ent.id,
                "ent_name": "",
                "mention_string": ent.mention.text,
                "cand": "" if these_cands is None else ",".join(these_cands['title'].unique()),
                "cand_i": -1,
                "cand_score": -1,
                "nil" : ent.nil,
                "comb_score" : -1,
                "comb_rank" : -1,

            }
            if ent.nil:
                self.query_tracing[ent.id]["source"] = "NIL"
        else:
            selected_sorted_candidates = {k: (v, i) for i, (k, v) in
                                 enumerate(sorted(ent.candidate_kb.items(), key=lambda item: item[1], reverse=True))}
            if ent.kbid in selected_sorted_candidates:
                self.query_tracing[ent.id]["comb_score"] = selected_sorted_candidates[ent.kbid][0]
                self.query_tracing[ent.id]["comb_rank"] = selected_sorted_candidates[ent.kbid][1]
                self.query_tracing[ent.id]["ent_name"] = ontology.entities[ent.kbid].name

            elif any(x for x in sorted_candidates.values() if x['subject'].replace("f_", "") == ent.kbid):
                val = [(i, x) for i, x in enumerate(sorted_candidates.values()) if x['subject'].replace("f_", "") == ent.kbid][0]
                self.query_tracing[ent.id]["comb_score"] = val[1]["_score"]
                self.query_tracing[ent.id]["comb_rank"] = val[0]
                self.query_tracing[ent.id]["ent_name"] = ""
            else:
                self.query_tracing[ent.id]["comb_score"] = -0.5
                self.query_tracing[ent.id]["comb_rank"] = -0.5
                self.query_tracing[ent.id]["ent_name"] = ""
        self.query_tracing[ent.id]["num_cands"] = len(ent.candidate_kb)

    def process_results(self, fbi, n_cands, query_string, candidates, ontology, num_hits = None, score_coeff = None, cand_rank = None, title=""):

        if num_hits is None:
            num_hits = math.ceil(n_cands * score_coeff)
        if query_string in self.cached_queries:
            results = self.cached_queries[query_string]
        else:
            results = fbi.retrieve(query_string, maxHits=num_hits)
            self.cached_queries[query_string] = results
        for r in results:
            r['_original_score'] = r['_score']
            # r['_cand_rank'] = cand_rank
            # r['_cand_score'] = score_coeff
            # r['_cand_title'] = title
            # if cand_rank is not None:
            #     r['_score'] = r['_score'] * (1. - float(cand_rank))
            # elif score_coeff is not None:
            #     r['_score'] = r['_score'] * score_coeff
            # r['_score'] = r['_score'] * math.log10(len(r))
            ent_name = ontology.get_name(r)
            sim = self.jw.normalized_similarity(title, ent_name)
            if score_coeff:
                r['_score'] = score_coeff * sim
            else:
                r['_score'] = sim
            # if sim < self.args.lex_min:
            #     continue
            # elif "r_type" in r and type(r["r_type"]) is list:
            #     skip = any(x for x in r['r_type'] if x in self.exclude_types)
            #     if skip:
            #         continue
            # elif "r_type" in r and type(r["r_type"]) is str:
            #     if r["r_type"] in self.exclude_types:
            #         continue
            if r['_docid'] not in candidates or candidates[r['_docid']]['_score'] < r['_score']:
                candidates[r['_docid']] = r
        return candidates
    def pull_candidates(self,ent, mention_candidates, fbi, ontology, args, n_cands = 200, exclude_nonwikis=False):
        these_cands = mention_candidates.loc[mention_candidates['id'] == ent.id]\
            .sort_values(by="score_ts", ascending=False).head(args.n_triage).reset_index()
        candidate_titles = set()
        tokens_before = [w for w in ent.mention.sent if w.i < ent.mention.start]
        skip = False
        if len(tokens_before) > 0:
            if "author=" in tokens_before[-1].text:
                skip = True
                ent.cand_skip = True
            elif "".join([w.text for w in ent.mention.sent if w.i < ent.mention.start]).endswith("author=\""):
                skip = True
                ent.cand_skip = True
        candidates = {}
        cached_candidates_ent = {}

        if not skip:
            for i, cand in these_cands.iterrows():
                if cand['title'] != 'NULLTITLE':
                    cand['title'] = re.sub(r"[^\w\d\s_]+", ' ', str(cand['title']))
                    if len(cand['title'].strip()) > 0:
                        query_string = "fk_key.wikipedia.en_title:\'{0}\' OR fk_key.wikipedia.en:\'{0}\'".format(cand['title'])
                        candidate_titles.add(cand['title'])
                        candidates = self.process_results(fbi, n_cands, query_string, candidates, ontology,
                                                          cand_rank=i,score_coeff=cand['score_ts'], title=cand['title'])
                        self.trace_gold(candidates, ent, query_type="wiki", this_cand=cand, this_cand_i=i)

            sorted_candidates = sorted([v for k, v in candidates.items()], key=lambda x : x['_score'], reverse=True)
            added_count = 0
            ent, cached_candidates_ent, added_count = self.add_candidates(
                sorted_candidates, exclude_nonwikis, added_count, ent, cached_candidates_ent, ontology, n_cands)

            candidates = {}
            if added_count < n_cands:
                for i, cand in these_cands.iterrows():
                    if cand['title'] != 'NULLTITLE':
                        cand['title'] = cand['title'] = re.sub(r"[^\w\d_\s]+", ' ',str(cand['title']))
                        if len(cand['title'].strip()) > 0:
                            query_string = cand['title'].replace("_", " ").replace("\n", " ")

                            try:
                                candidates = self.process_results(fbi, n_cands, query_string, candidates, ontology, cand_rank=i,
                                                                  score_coeff=cand['score_ts'], title=cand['title'])
                                self.trace_gold(candidates, ent, query_type="wiki_all", this_cand=cand, this_cand_i=i)

                            except Exception as e:
                                logging.getLogger().info("error - Title:{0}, mention:{1}".format(query_string, cand['id']))
                                logging.getLogger().error(str(e))
                                logging.getLogger().error(traceback.format_exc())

                if not re.search(u'[\u4e00-\u9fff]', ent.mention.text):
                    query_string = re.sub(r"[^\w\d_\s]+", ' ', ent.mention.text)
                    if len(query_string.strip()) > 0:
                        candidates = self.process_results(fbi, n_cands, query_string, candidates, ontology, num_hits=20,
                                                          title= ent.mention.text)
                        self.trace_gold(candidates, ent, query_type="string")

            sorted_candidates = sorted([v for k, v in candidates.items()], key=lambda x : x['_score'], reverse=True)
            ent, cached_candidates_ent, added_count = self.add_candidates(
                sorted_candidates, exclude_nonwikis, added_count, ent, cached_candidates_ent, ontology, n_cands)
        self.trace_end(ent, these_cands, candidates, ontology)
        return ent, ontology, candidate_titles, cached_candidates_ent

    def add_candidates(self,sorted_candidates, exclude_nonwikis, added_count, ent, cached_candidates_ent, ontology, n_cands):
        excluded_fields = [
        ]

        for data in sorted_candidates:
            skip = False
            if added_count >= n_cands:
                skip = True
            if exclude_nonwikis and not any(x for x in data if 'fk_key.wikipedia' in x):
                skip = True
            for data_field in data:
                for ef in excluded_fields:
                    if ef in data_field:
                        skip = True
                        break
            if not skip:
                data["subject"] = data["subject"].replace("f_", "")
                added = ontology.add(data["subject"].replace("f_", ""), data, add_if_no_name=True)
                if added:
                    added_count += 1
                    ent.add_candidate(data["subject"], data['_score'])
                    cached_candidates_ent[data["subject"]] = data
        return ent, cached_candidates_ent, added_count

    def pull_candidates_cached(self,ent, cached_candidates, ontology, n_cands = 200, exclude_nonwikis=False):

        added_count = 0

        for cand_id, data in cached_candidates[ent.id].items():
            if added_count >= n_cands:
                break
            if exclude_nonwikis and not any(x for x in data if 'fk_key.wikipedia' in x):
                break
            added = ontology.add(data["subject"].replace("f_", ""), data, add_if_no_name=False)
            if added:
                added_count += 1
                ent.add_candidate(data["subject"], data['_score'])


        return ent, ontology