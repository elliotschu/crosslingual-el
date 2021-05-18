"""
Elliot Schumacher, Johns Hopkins University
Created 6/11/20
"""
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
from scipy.stats import describe
import textdistance
import math
from data.Objects import Ontology
def main():
    timestamp = "run_2020_06_11_16_33_10_ALL_8548564_r8n37"

    included_rank_list = []
    nil_list = []
    ont = Ontology()

    exclude_set = set()
    type_exclude_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/non_gold_types.txt"
    with open(type_exclude_file) as f:
        for line in f:
            exclude_set.add(line.strip())

    included_set = set()
    type_include_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/gold_types.txt"
    with open(type_include_file) as f:
        for line in f:
            included_set.add(line.strip())

    all_types = set()
    gold_types = set()

    data_file = "/Users/elliotschumacher/Dropbox/git/clel/datasets/LDC2019T02/tac_kbp_2015_tedl_training_gold_standard_entity_mentions.tab"
    header_list = ["system_id", "query_id", "mention_string", "doc_id_offsets", "link_id", "entity_type",
                   "mention_type",
                   "confidence", "web_search", "wiki_text", "unknown"]
    data = pd.read_csv(data_file, sep="\t", names=header_list, header=None, )

    for lang in ["SPA", "CMN", "ENG"]:

        tracing_csv_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/tracingtraining{lang}.csv"
        tracing = pd.read_csv(tracing_csv_file)
        cand_file = f"/Users/elliotschumacher/Dropbox/git/clel/datasets/LDC2019T02/{lang}/mentions_{lang}_out.csv"
        cand_df = pd.read_csv(cand_file)

        candidate_pkl_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/{lang}_candidates_LDC2019T02_sim4.shrink.sort.index_200.pkl"


        with open(candidate_pkl_file, 'rb') as cand_pkl:
            cands = pickle.load(cand_pkl)


        for i, row in data.iterrows():
            if row.doc_id_offsets.startswith(lang):
                these_cands = cands[row.query_id]
                these_trace = tracing[tracing['ent_id'] == row.query_id]
                #these_cands_df = cand_df[cand_df['id'] == row.query_id]
                these_new_cands = {}
                for tc in these_cands:
                    if ("r_type" not in these_cands[tc]) or \
                            (type(these_cands[tc]['r_type']) is str and these_cands[tc]['r_type'] not in exclude_set) \
                            or (type(these_cands[tc]['r_type']) is list and all(x for x in these_cands[tc]['r_type'] if x not in exclude_set)):
                        #or (type(these_cands[tc]['r_type']) is list and all(x for x in these_cands[tc]['r_type'] if x in included_set)):
                        these_new_cands[tc] = these_cands[tc]
                        these_new_cands[tc]['kbid'] = tc
                        these_new_cands[tc]['ont_name'] = ont.get_name(these_cands[tc])
                    if "r_type" in  these_cands[tc]:
                        if type(these_cands[tc]["r_type"]) is list:
                            all_types.update(these_cands[tc]["r_type"])
                        else:
                            all_types.add(these_cands[tc]["r_type"])

                df_cands = pd.DataFrame.from_dict(these_new_cands.values())
                # if "r_type" in  df_cands.columns:
                #     all_types.update(set(x for x in df_cands['r_type'][df_cands['r_type'].notnull()].values if type(x) is str))
                #     all_types.update(set(x for x_list in df_cands['r_type'][df_cands['r_type'].notnull()].values for x in x_list if type(x_list) is list))
                if len(df_cands) > 0:
                    dist = textdistance.JaroWinkler()

                    df_cands['_inv_rank'] = (1.0 / (df_cands['_cand_rank'] + 1))
                    df_cands['_combined'] = df_cands['_original_score'] * df_cands['_cand_score']
                    df_cands['_combined_inv'] = df_cands['_original_score'] * df_cands['_inv_rank']
                    df_cands['_lex_dist'] = df_cands.apply(lambda x: dist.normalized_similarity(x._cand_title, x.ont_name), axis=1)
                    df_cands['_cand_length'] = df_cands.apply(lambda x: len(x._cand_title), axis=1)
                    df_cands['_length'] = df_cands.count(axis=1)
                    df_cands['_tri_lex'] = df_cands['_lex_dist'] * df_cands['_cand_score']

                    df_cands = df_cands[(df_cands['_lex_dist'] >= 0.6) | (df_cands['_cand_length'] == 0)]


                if len(these_cands) > 0 and not row.link_id.startswith("NIL") and row.link_id in df_cands['kbid'].values:
                    if "r_type" in these_cands[row.link_id]:
                        if type(these_cands[row.link_id]["r_type"]) is list:
                            ent_type = these_cands[row.link_id]["r_type"]
                        else:
                            ent_type = [these_cands[row.link_id]["r_type"]]
                        gold_types.update(ent_type)

                    ranks = {}

                    sorting_keys = [["_cand_score", "_original_score"],
                                    "_cand_score",
                                    "_lex_dist",
                                    "_length",
                                    ["_cand_score", "_lex_dist"],
                                    ["_cand_score", "_lex_dist", '_length'],
                                    ["_cand_score",'_length'],
                                    "_tri_lex"]
                    for sk in sorting_keys:
                        this_sorted = df_cands.sort_values(by=sk, ascending=False).reset_index()
                        str_sk = sk
                        if type(sk) is list:
                            str_sk = ",".join(sk)
                        ranks[str_sk] = this_sorted[this_sorted['kbid'] == row.link_id].head(1).index.values[0]
                    ranks["orig_length"] = len(these_cands)
                    ranks["sort_length"] = len(df_cands)
                    ranks["_lex_dist_score"] = df_cands[df_cands['kbid'] == row.link_id].head(1)['_lex_dist'].iloc[0]
                    if these_trace['source'].iloc[0] != "string":
                        ranks["_cand_i"] = these_trace['cand_i'].iloc[0]
                    included_rank_list.append(ranks)

                elif row.link_id.startswith("NIL"):
                    info = {}
                    info["nil_num"] = len(df_cands)
                    if info["nil_num"] > 0:
                        for measure in ['_lex_dist', "_cand_score"]:
                            if not math.isnan(df_cands[measure].max()):
                                vals = df_cands[df_cands[measure].notna()][measure]
                                info[f"{measure}_max_score"] = vals.max()
                                info[f"{measure}_min_score"] = vals.min()
                                info[f"{measure}_mean_score"] = vals.mean()
                                info[f"{measure}_std_score"] =vals.std()
                            else:
                                info[f"{measure}_max_score"] = 0.
                                info[f"{measure}_min_score"] = 0.
                                info[f"{measure}_mean_score"] = 0.
                                info[f"{measure}_std_score"] = 0.
                    else:
                        for measure in ['_lex_dist', "_cand_score"]:

                            info[f"{measure}_max_score"] = 0.
                            info[f"{measure}_min_score"] = 0.
                            info[f"{measure}_mean_score"] = 0.
                            info[f"{measure}_std_score"] = 0.
                    nil_list.append(info)

    rank_df = pd.DataFrame.from_dict(included_rank_list)

    all_types_f =  f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/all_types.txt"
    gold_types_f =  f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/gold_types.txt"
    non_gold_types_f =  f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/non_gold_types.txt"
    with open(all_types_f, 'w') as f:
        sorted_types = sorted(all_types)
        for t in sorted_types:
            f.write(t + "\n")

    with open(gold_types_f, 'w') as f:
        sorted_types = sorted(gold_types)
        for t in sorted_types:
            f.write(t + "\n")

    with open(non_gold_types_f, 'w') as f:
        sorted_types = sorted(all_types - gold_types)
        for t in sorted_types:
            f.write(t + "\n")

    for col in rank_df.columns:
        print(col)
        print(describe(rank_df[col]))

        if col == '_lex_dist_score':
            for k in np.linspace(0,1,11):
                num_in_k = rank_df[rank_df[col] >= (1.0 - k)]
                k_perc = len(num_in_k) / float(len(rank_df))
                print(f"k={1.0 - k}, {k_perc}, {len(num_in_k)}")
        elif col == '_cand_i':
            print(describe(rank_df[rank_df['_cand_i'].notna()]['_cand_i']))
            for k in range(0,10):
                num_in_k = rank_df[rank_df[col] <= k]
                k_perc = len(num_in_k) / float(len(rank_df))
                print(f"k={k}, {k_perc}, {len(num_in_k)}")
        else:
            for k in [5, 10, 25, 50, 100, 150, 200]:
                num_in_k = rank_df[rank_df[col] < k]
                k_perc = len(num_in_k) / float(len(rank_df))
                print(f"k={k}, {k_perc}, {len(num_in_k)}")
    nil_df = pd.DataFrame.from_dict(nil_list)

    for col in nil_df.columns:
        print(col)
        print(describe(nil_df[col]))

if __name__ == "__main__":
    main()