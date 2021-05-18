import numpy as np
import logging
import csv
import scipy.stats as st
from codebase import mention_links
from collections import defaultdict
import pandas as pd
from codebase import el_scorer
import os
"""
 from Spotlight codebase (see https://github.com/maciejkula/spotlight)
"""

FLOAT_MAX = np.finfo(np.float32).max
def tac_script(tac_result_list, epoch_num, outpath, args):
    os.makedirs(os.path.join(outpath, 'eval_{0}'.format(epoch_num)), exist_ok=True)

    threshold_array = np.linspace(-1,1,args.threshold_num)
    logging.getLogger().info("Thresholds:{0}".format(threshold_array))
    for this_threshold in threshold_array:
        pred_filename = os.path.join(outpath, 'eval_{0}'.format(epoch_num), "t_{0:.2f}.pred.tac".format(this_threshold))
        with open(pred_filename, 'w') as tac_f, \
                open(os.path.join(outpath,"eval.gold.tac"), 'w') as tac_gold_f:
            logging.getLogger().info("Threshold score {0}".format(this_threshold))
            for r_dict in tac_result_list:
                if r_dict["score"] >= this_threshold:
                    tac_f.write("{0} {1} {2}\n".format(r_dict["mention_id"], r_dict["pred_kbid"]["concept_id"], r_dict["score"]))
                else:
                    tac_f.write("{0} NIL {1}\n".format(r_dict["mention_id"], r_dict["score"]))
                tac_gold_f.write("{0} {1}\n".format(r_dict["mention_id"], r_dict["gold_kbid"]))
    all_results = el_scorer.systemsRankingScript(goldStdFile= os.path.join(outpath,"eval.gold.tac"),
                                   systemsDir= os.path.join(outpath, 'eval_{0}'.format(epoch_num)),
                                   focusElFile= os.path.join(outpath,"eval.gold.tac"))

    results_df = pd.DataFrame.from_dict(all_results)
    results_df.sort_values('system_name')
    results_df.to_csv(os.path.join(outpath, '{0}_res.csv'.format(epoch_num)), index=False,)

    if args.language.upper() == "ALL":
        language_set = set(x["language"] for x in tac_result_list)
        for lang in language_set:
            os.makedirs(os.path.join(outpath, 'eval_{0}'.format(epoch_num), lang), exist_ok=True)

            threshold_array = np.linspace(-1, 1, args.threshold_num)
            logging.getLogger().info("Thresholds:{0}".format(threshold_array))
            for this_threshold in threshold_array:
                pred_filename = os.path.join(outpath, 'eval_{0}'.format(epoch_num), lang,
                                             "t_{0:.2f}.pred.tac".format(this_threshold))
                with open(pred_filename, 'w') as tac_f, \
                        open(os.path.join(outpath, "eval_{0}.gold.tac".format(lang)), 'w') as tac_gold_f:
                    for r_dict in tac_result_list:
                        if r_dict["language"] == lang:
                            if r_dict["score"] >= this_threshold:
                                tac_f.write("{0} {1} {2}\n".format(r_dict["mention_id"], r_dict["pred_kbid"]["concept_id"],
                                                                   r_dict["score"]))
                            else:
                                tac_f.write("{0} NIL {1}\n".format(r_dict["mention_id"], r_dict["score"]))
                            tac_gold_f.write("{0} {1}\n".format(r_dict["mention_id"], r_dict["gold_kbid"]))
            lang_results = el_scorer.systemsRankingScript(goldStdFile=os.path.join(outpath, "eval_{0}.gold.tac".format(lang)),
                                                         systemsDir=os.path.join(outpath, 'eval_{0}'.format(epoch_num), lang),
                                                         focusElFile=os.path.join(outpath,"eval_{0}.gold.tac".format(lang)))

            results_df = pd.DataFrame.from_dict(lang_results)
            logging.getLogger().info(list(results_df.columns))
            results_df.to_csv(os.path.join(outpath, '{0}_{1}_res.csv'.format(epoch_num, lang)), index=False, )

    return all_results

def resolve_tie(mention_links, predictions, max_score_indxs):
    return np.random.choice(max_score_indxs[0])
    pass


def score(mention_links, predictions, test_dict, outpath=None, output_top=10):
    """
    Given a set of mention ids with an attached devevelopment set, scores all instances in the development set and outputs
    appropriate logging information to a sub-directory of the result root directory.
    :param mention_links: a mention links object containing a development (test) set of mentions
    :param predictions: predictions for each mention
    :param test_dict: a dictionary containing the original documents and annotations for the testing partition.
    :param outpath:
    :param output_top: top n scores to return.
    :return:
    """
    log = logging.getLogger()
    score_list = defaultdict(lambda: [])
    score_list_lang = defaultdict(lambda: defaultdict(lambda: []))

    result_list = []
    tac_result_list = []
    ties = 0
    for indx, mention_id in enumerate(mention_links.test_mention_ids):

        mention = [mention for mention in test_dict.entities if mention.id == mention_id][0]


        candidate_index_list = []
        candidate_index_map = {}
        candidate_index_map_inverse = {}

        for i, cand_kbid in enumerate(mention.candidate_kb.keys()):
            if cand_kbid in mention_links.cui_to_concept_info:
                candidate_index_list.append([x["index"] for x in mention_links.cui_to_concept_info[cand_kbid]][0])
                real_indx = [x["index"] for x in mention_links.cui_to_concept_info[cand_kbid]][0]
                candidate_index_map[i] = real_indx
                candidate_index_map_inverse[real_indx] = i

        predictions[indx, [q for q in range(predictions.shape[1]) if q not in candidate_index_map.keys()]] = np.float64(
            "-inf")
        ranking = st.rankdata(-predictions[indx, :])
        max_ranking = st.rankdata(-predictions[indx, :], method='max')
        min_ranking = st.rankdata(-predictions[indx, :], method='min')
        max_score_indxs = np.where(predictions[indx, :] == predictions[indx, :].max())
        if len(max_score_indxs[0]) == 1:
            converted_index = candidate_index_map[max_score_indxs[0][0]]
            tac_output = {
                "mention_id" : mention.id,
                "score" : predictions[indx, max_score_indxs[0][0]],
                "pred_kbid" : mention_links.id_to_concept_info[converted_index],
                "inc_thres" : "NIL" not in mention.kbid.upper(),
                "gold_kbid" : mention.kbid,
                "language" : mention.lang,
             }
        elif len(candidate_index_list) > 0 and len(max_score_indxs[0]) > 1:
            try:
                ties += 1
                pred_index = resolve_tie(mention_links, predictions, max_score_indxs)
                converted_index = candidate_index_map[pred_index]

                tac_output = {
                    "mention_id": mention.id,
                    "score": predictions[indx, pred_index],
                    "pred_kbid": mention_links.id_to_concept_info[converted_index],
                    "inc_thres": "NIL" not in mention.kbid.upper(),
                    "gold_kbid": mention.kbid,
                    "language": mention.lang,

                }
            except Exception as e:
                print(mention)
                print(e)
                print(max_score_indxs)
                print(pred_index)
        else:

            tac_output = {
                "mention_id": mention.id,
                "score": np.float64("-inf"),
                "pred_kbid": "NIL",
                "inc_thres": "NIL" not in mention.kbid.upper(),
                "gold_kbid": mention.kbid,
                "language": mention.lang,

            }
        tac_result_list.append(tac_output)

        if "NIL" not in mention.kbid.upper():
            if mention.kbid in mention.candidate_kb.keys() and mention.kbid in mention_links.cui_to_concept_info:
                gold_concept_indx = [candidate_index_map_inverse[x["index"]] for x in mention_links.cui_to_concept_info[mention.kbid]]

                gold_rank_indx = gold_concept_indx[ranking[gold_concept_indx].argmin()]
                gold_rank = ranking[gold_rank_indx]

                gold_rank_max_indx = gold_concept_indx[max_ranking[gold_concept_indx].argmin()]
                gold_rank_max = max_ranking[gold_rank_max_indx]

                gold_rank_min_indx = gold_concept_indx[min_ranking[gold_concept_indx].argmin()]
                gold_rank_min = min_ranking[gold_rank_min_indx]

                score = predictions[indx, gold_concept_indx].max()
                score_list["mrr"].append(1.0 / gold_rank)
                score_list["max_mrr"].append((1.0 / gold_rank_max))
                score_list["min_mrr"].append((1.0 / gold_rank_min))
                score_list["no_cands"].append(0.)
                score_list["accuracy"].append(1. if gold_rank == np.float64(1) else 0.)
                score_list["max_accuracy"].append(1. if gold_rank_max == np.float64(1) else 0.)
                score_list["min_accuracy"].append(1. if gold_rank_min == np.float64(1) else 0.)

                score_list_lang[mention.lang]["mrr"].append(1.0 / gold_rank)
                score_list_lang[mention.lang]["max_mrr"].append((1.0 / gold_rank_max))
                score_list_lang[mention.lang]["min_mrr"].append((1.0 / gold_rank_min))
                score_list_lang[mention.lang]["no_cands"].append(0.)
                score_list_lang[mention.lang]["accuracy"].append(1. if gold_rank == np.float64(1) else 0.)
                score_list_lang[mention.lang]["max_accuracy"].append(1. if gold_rank_max == np.float64(1) else 0.)
                score_list_lang[mention.lang]["min_accuracy"].append(1. if gold_rank_min == np.float64(1) else 0.)


            else:
                score_list["mrr"].append(0.)
                score_list["max_mrr"].append(0.)
                score_list["min_mrr"].append(0.)
                score_list["no_cands"].append(1.)
                score_list["accuracy"].append(0.)
                score_list["max_accuracy"].append(0.)
                score_list["min_accuracy"].append(0.)

                score_list_lang[mention.lang]["mrr"].append(0.)
                score_list_lang[mention.lang]["max_mrr"].append(0.)
                score_list_lang[mention.lang]["min_mrr"].append(0.)
                score_list_lang[mention.lang]["no_cands"].append(1.)
                score_list_lang[mention.lang]["accuracy"].append(0.)
                score_list_lang[mention.lang]["max_accuracy"].append(0.)
                score_list_lang[mention.lang]["min_accuracy"].append(0.)


        else:
            gold_rank = ""
            score = float("-inf")


        if outpath:
            top_ind = np.argpartition(predictions[indx, :],-output_top)[-output_top:]
            pred_list = []
            for q in top_ind:
                if q in candidate_index_map:
                    try:
                        q_conv = candidate_index_map[q]
                        pred_list.append((mention_links.id_to_concept_info[q_conv]["concept_id"],
                                          mention_links.id_to_concept_info[q_conv]["name"],
                                          predictions[indx, q],
                                          ranking[q]))
                    except:
                        log.info(mention)
            pred_list = sorted(pred_list, key=lambda x: x[-1])
            try:
                sentence = str(mention.mention.sent)
            except:
                log.warning("Cannot print sentence for mention:{0}".format(mention.id))
                sentence = ""

            row = {
                "_text": " ".join([w.text for w in mention.mention]),
                "_sentence" : sentence,
                "_gold_kbid": mention.kbid,
                "_gold_cui_rank" : gold_rank,
                "_gold_cui_score" : score,
                "_num_kb_cands": len(mention.candidate_kb),
                "~~mention_uuid": mention.id,
                "~~comm": mention.doc_filename,
            }

            for ip, pred in enumerate(pred_list):
                row["~pred_cuis_{0}".format(ip)] = "{0}={1} ({2:.2f})".format(pred[0], pred[1], pred[2])
            if "NIL" not in mention.kbid.upper():

                for fn in score_list:
                    row[fn] = score_list[fn][-1]
                row["_gold_name"] = mention_links.cui_to_concept_info[mention.kbid][0]["name"]
            else:
                row["_gold_name"] = "NIL"

            result_list.append(row)
    tie_perc = ties / float(len(result_list))
    log.info("Percentage with ties:{0:.2f}".format(tie_perc))

    results = {
        "mrr": np.mean(score_list["mrr"]),
        "max_mrr" : np.mean(score_list["max_mrr"]),
        "min_mrr": np.mean(score_list["min_mrr"]),
        "accuracy" : np.mean(score_list["accuracy"]),
        "max_accuracy": np.mean(score_list["max_accuracy"]),
        "min_accuracy": np.mean(score_list["min_accuracy"]),
        "no_cands_perc": np.mean(score_list["no_cands"]),

    }
    if outpath:
        options = {}
        options['strings_to_formulas'] = False
        options['strings_to_urls'] = False
        writer = pd.ExcelWriter(outpath + '.xlsx', engine='xlsxwriter', options=options)

        dataframe = pd.DataFrame.from_dict(result_list)
        dataframe.to_excel(writer, index=False, freeze_panes=(1,0))
        writer.save()



    return results, tac_result_list
